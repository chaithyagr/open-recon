
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import ctypes
import mrdhelper
from datetime import datetime

# Folder for debug output files
import os
import numpy as np
import ismrmrd
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ggrappa.grappaND import GRAPPA_Recon, estimate_grappa_kernel
import torch

from functools import partial
import itertools
import sys
import scipy as sp
from mrinufft.io.nsp import read_trajectory, DEFAULT_RASTER_TIME
from mrinufft import get_density, get_operator
import warnings
from ggrappa.grappaND import GRAPPA_Recon
import torch
from ggrappa.utils import get_cart_portion_sparkling, get_grappa_filled_data_and_loc

def do_grappa_and_append_data(kspace_loc, kspace_data, traj_params, grappa_maker, acs=None, recon_hw="cpu"):
    kspace_shots = kspace_loc.reshape(traj_params['num_shots'], -1, traj_params['dimension'])
    gridded_center, new_kspace_data, new_kspace_loc = get_cart_portion_sparkling(kspace_shots, traj_params, kspace_data)
    if acs is not None:
        if acs.shape[1] != traj_params['img_size'][0]:
            warnings.warn("ACS size does not match the image size. Re-sampling")
            acs = sp.signal.resample(
                acs, traj_params['img_size'][0], axis=1
            )
    grappa_recon, grappa_kernel = grappa_maker(
        sig=torch.tensor(gridded_center).permute(0, 2, 3, 1),
        acs=torch.tensor(acs).permute(0, 2, 3, 1) if acs is not None else None,
        isGolfSparks=True,
        cuda=recon_hw=="gpu",
    )
    grappa_recon = grappa_recon.permute(0, 3, 1, 2).numpy()
    extra_loc, extra_data = get_grappa_filled_data_and_loc(gridded_center, grappa_recon, traj_params)
    kspace_loc = np.concatenate([new_kspace_loc, extra_loc], axis=0)
    kspace_data = np.hstack([new_kspace_data, extra_data])
    return kspace_loc, kspace_data

debugFolder = "/tmp/share/debug"

def get_user_param(param_list, name):
    """
    Get a parameter from a list of parameters
    """
    for param in param_list:
        if param.name == name:
            return param.value
    return None

def peek_acquisition(acquisition):
    acq = next(acquisition)
    acquisition = itertools.chain([acq], acquisition)
    return acq, acquisition


def online_phase_shift(acq, kspace_loc, shifts, fov, vol_shape):
    if acq is None:
        return None
    kspace_data = acq.data
    if np.sum(shifts) == 0:
        return kspace_data
    normalized_shift = np.array(shifts) / np.array(fov) * np.array(vol_shape)
    phi = np.exp(-2 * np.pi * 1j * np.sum(kspace_loc * normalized_shift, axis=-1))
    return kspace_data * phi


def accumulate_and_shift_acs(acquisition):
    max_phase_enc = 0
    max_part_enc = 0
    acs_data = []
    for acq in acquisition:
        if acq is None:
            break
        elif acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
            if acq.idx.kspace_encode_step_1 > max_phase_enc:
                max_phase_enc = acq.idx.kspace_encode_step_1 + 1
            if acq.idx.kspace_encode_step_2 > max_part_enc:
                max_part_enc = acq.idx.kspace_encode_step_2 + 1
            acs_data.append(acq.data)
        else:
            break
    acs_data = np.stack(acs_data, axis=-1)
    acs_data = acs_data.reshape(*acs_data.shape[:2], max_phase_enc, max_part_enc)
    acs_data = np.swapaxes(acs_data, -1, -2)
    return acs_data, itertools.chain([acq], acquisition)


def phase_shift_and_accumulate_kspace(acquisition, phi=None, reshape=True, accumulate=100):
    def _return_kspace(data):
        kspace_data = np.hstack(data)
        if reshape:
            return kspace_data.reshape(data[0].shape[0], -1), traj_counter
        else:
            return kspace_data, traj_counter
    traj_counter = 0
    counter = 0
    kspace_data = []
    for acq in acquisition:
        if isinstance(acq, ismrmrd.acquisition.Acquisition):
            data = acq.data
        else:
            data = acq
        if phi is not None:
            data *= phi[traj_counter]
            traj_counter += 1
        kspace_data.append(data)
        counter += 1
        if accumulate != 'all' and counter == accumulate:
            yield _return_kspace(kspace_data)
            counter = 0
            kspace_data = []
    if accumulate == 'all' or counter != accumulate:
        yield _return_kspace(kspace_data)


def get_eigvecs(kspace_data, coil_out=5):
    square_arr = kspace_data.conj() @ kspace_data.T
    _, eigvecs = sp.linalg.eigh(square_arr)
    #compressed_kspace = eigvecs[:, -coil_out:][:, ::-1].T @ kspace_data
    return eigvecs[:, -coil_out:][:, ::-1]


class Accumulator:
    def __init__(self, total_data, accumulate=100, reshape=True, phi=None):
        self.accumulate = accumulate
        self.reshape = reshape
        self.total_data = total_data
        self.counter = 0
        self.traj_counter = 0
        self.phi = None
        self.accumulated_kspace = []
    
    def _return_kspace(self, data):
        kspace_data = np.hstack(data)
        self.counter = 0
        self.accumulated_kspace = []
        if self.reshape:
            return kspace_data.reshape(kspace_data.shape[0], -1), self.traj_counter
        else:
            return kspace_data, self.traj_counter
        
    def accumulate_data(self, acq):
        if isinstance(acq, ismrmrd.acquisition.Acquisition):
            data = acq.data
        else:
            data = acq
            if self.phi is not None:
                data *= self.phi[self.traj_counter]
                self.traj_counter += 1
        self.accumulated_kspace.append(data)
        self.counter += 1
        if self.accumulate != 'all' and self.counter == self.accumulate:
            return self._return_kspace(self.accumulated_kspace)
        if self.accumulate == 'all' or self.traj_counter == self.total_data:
            return self._return_kspace(self.accumulated_kspace)
        return None

def yield_return(generator_in, out_every=100):
    for i, e in enumerate(generator_in):
        if not i%out_every:
            yield e
     
    
class OnlineCompresser:
    def __init__(self, coil_out=5):
        self.coil_out = coil_out
        self.eigvects = None
         
    def compress_coils(self, kspace_data, traj_counter):
        #print("Compressing coils for ", traj_counter)
        st = time.time()
        if self.coil_out >= kspace_data.shape[0]:
            compressed_kspace = kspace_data
        else:
            if self.eigvects is None:
                self.eigvects = get_eigvecs(kspace_data, self.coil_out)
            compressed_kspace = self.eigvects.T @ kspace_data
        #print("Time for compressing coils at traj:",traj_counter, " :: ", time.time() - st)
        return compressed_kspace
    

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)
    logging.info("MRD Header: \n%s", mrdHeader)

    ParLong     = mrdHeader.userParameters.userParameterLong
    ParDouble   = mrdHeader.userParameters.userParameterDouble
    ParString   = mrdHeader.userParameters.userParameterString

    # Extract some useful information for reconstruction
    is3d        = (mrdHeader.encoding[0].encodingLimits.kspace_encoding_step_2.maximum>0)

    NoOfSlice   = mrdHeader.encoding[0].reconSpace.matrixSize.z if is3d else mrdHeader.encoding[0].encodingLimits.slice.maximum
    NoOfSpokes  = mrdHeader.encoding[0].encodingLimits.kspace_encoding_step_1.maximum+1

    RawMatX     = mrdHeader.encoding[0].encodedSpace.matrixSize.x
    RawMatY     = mrdHeader.encoding[0].encodedSpace.matrixSize.y
    RecoMatX    = mrdHeader.encoding[0].reconSpace.matrixSize.x
    RecoMatY    = mrdHeader.encoding[0].reconSpace.matrixSize.y

    try:
        recon_hw = config['parameters']['reconhw']
    except:
        recon_hw = "cpu"

    fov = (
        mrdHeader.encoding[0].reconSpace.fieldOfView_mm.x / 1000,
        mrdHeader.encoding[0].reconSpace.fieldOfView_mm.y / 1000,
        mrdHeader.encoding[0].reconSpace.fieldOfView_mm.z / 1000,
    )
    OSFactor            = get_user_param(ParLong, 'OversamplingFactor')
    NoOfReadoutSamples  = get_user_param(ParLong, 'NumberOfReadoutSamples')
    LoadGradientFile    = True #get_user_param(ParLong, 'SendTrajectory') == 1
    turbo_factor         = get_user_param(ParLong, 'TurboFactor')
    fov_shift = tuple([
        get_user_param(ParDouble, shift)/1000 
        for shift in ['ShiftInReadout', 'ShiftInPhase', 'ShiftInSlice']
    ])
    vol_shape = (RawMatX, RawMatY, NoOfSlice)
    # vol_shape = (256, 256, 176)
    # fov = (0.256, 0.256, 0.176)

    Kmax = np.array(vol_shape)/2/np.array(fov)

    logging.info("--->")
    logging.info("Protocol name    : "+str(mrdHeader.measurementInformation.protocolName))
    logging.info("3D protocol      : "+str(is3d))
    logging.info("Raw matrix       : "+str(RawMatX)+"x"+str(RawMatY))
    logging.info("Recon matrix     : "+str(RecoMatX)+"x"+str(RecoMatY))
    logging.info("Recon volume     : " + str(vol_shape))
    logging.info("FOV Size         : " + str(fov))
    logging.info("Kmax         : " + str(Kmax))
    logging.info("Num. of slices   : "+str(NoOfSlice))
    logging.info("Num. of spokes   : "+str(NoOfSpokes))
    logging.info("Num. of samples  : "+str(NoOfReadoutSamples))
    logging.info("OS factor        : "+str(OSFactor))
    logging.info("Recon HW        : "+str(recon_hw))
    # ------ printing user parameters -----
    logging.info("User parameters long")
    for k in range(len(ParLong)):
        logging.info(str(k)+" -> "+ParLong[k].name+" = "+str(ParLong[k].value))
    logging.info("User parameters double")
    for k in range(len(ParDouble)):
        logging.info(str(k)+" -> "+ParDouble[k].name+" = "+str(ParDouble[k].value))
    logging.info("User parameters string")
    for k in range(len(ParString)):
        logging.info(str(k)+" -> "+ParString[k].name+" = "+ParString[k].value)
    logging.info("<---")

    trajectory, traj_params = read_trajectory(
        os.path.join('/opt/code/python-ismrmrd-server/data', get_user_param(ParString, 'GradientFilename')),
        dwell_time=DEFAULT_RASTER_TIME/OSFactor,
        num_adc_samples=int(NoOfReadoutSamples),
    )
    trajectory = np.clip(trajectory, -0.5, 0.5)
    kspace_loc = trajectory.reshape(-1, trajectory.shape[-1]).astype(np.float32)
    
    if turbo_factor > 1:
        num_inversions = NoOfSpokes // turbo_factor + 1
        reorder_pos = np.arange(num_inversions * turbo_factor).reshape(turbo_factor, num_inversions).T.flatten()
        reorder_pos = np.delete(reorder_pos, reorder_pos>=NoOfSpokes)
        trajectory = trajectory[reorder_pos]
        kspace_loc = trajectory.reshape(-1, trajectory.shape[-1]).astype(np.float32)
    
    try:
        # Remove this and add it inside the gradient file
        af_string = get_user_param(ParString, 'GradientFilename').split('_G')[1].split('_')[0].split('x')
        delta = 0
        if len(af_string) > 1 and 'd' in af_string[1]:
            af_caipi = af_string[1].split('d')
            af_string[1] = af_caipi[0]
            if int(af_caipi[1])>0:
                delta = int(af_caipi[1])
        af = tuple([int(float(af)) for af in af_string])
    except:
        af = (1,)
    
    acq, acquisition = peek_acquisition(iter(connection))
    grappa_recon_kernels = None
    if acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
        acs_data, acquisition = accumulate_and_shift_acs(acquisition)
        if acs_data.shape[1] != vol_shape[0]:
            warnings.warn("ACS size does not match the image size. Re-sampling")
            acs_data = sp.signal.resample(
                acs_data, traj_params['img_size'][0], axis=1
            )
        grappa_recon_kernels = estimate_grappa_kernel(acs=torch.from_numpy(acs_data).permute(0, 2, 3, 1), af=af, delta=delta)
    kspace_gen = Parallel(n_jobs=1, return_as='generator', verbose=1000)(
        delayed(online_phase_shift)
        (acq, trajectory[traj_num] if acq is not None else None, fov_shift, fov, vol_shape)
        for traj_num, acq in enumerate(acquisition)
    )
    if np.prod(af) == 1:
        density_comp = get_density("pipe")(kspace_loc, vol_shape)
    else:
        density_comp = True
    kspace_data = np.stack([kspace for kspace in kspace_gen][:-1], axis=1) # Exclude the last None, only in Open Recon
    kspace_data = np.ascontiguousarray(kspace_data.reshape(kspace_data.shape[0], -1))
    if np.prod(af) > 1:
        grappa_reconstructor = partial(GRAPPA_Recon, grappa_recon_spec=grappa_recon_kernels)
        kspace_loc, kspace_data = do_grappa_and_append_data(kspace_loc, kspace_data, traj_params, grappa_reconstructor, recon_hw=recon_hw)
    fourier_op = get_operator("finufft" if recon_hw == "cpu" else "gpunufft")(
        kspace_loc.astype(np.float32),
        vol_shape,
        n_coils=kspace_data.shape[0],
        density=density_comp,
        smaps={"name": "low_frequency", "kspace_data": kspace_data, "max_iter": 1},
    )
    recon_method = "adjoint"
    img = fourier_op.adj_op(kspace_data)    
    img = np.abs(img)
    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored")
    maxVal = 2**BitsStored - 1
    # Normalize and convert to int16
    img *= maxVal/img.max()
    img = np.around(img).astype(np.int16)

    send_as_3d_image = False

    if send_as_3d_image:
        # Format as ISMRMRD image data
        # data has shape [RO PE], i.e. [x y].
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        image = ismrmrd.Image.from_array(img.transpose(), transpose=False)
        image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), acq.getHead()))

        image.image_index = 1

        # Set field of view
        image.field_of_view = (
            ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.x), 
            ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.y), 
            ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.z)
        )
    
        # Set ISMRMRD Meta Attributes
        meta = ismrmrd.Meta()
        meta['DataRole']                       = 'Image'
        meta['ImageProcessingHistory']         = ['OPENRECON', 'PYTHON']
        meta['SequenceDescriptionAdditional']  = 'OPENRECON'
        meta['Keep_image_geometry']            = 1

        # Add image orientation directions to MetaAttributes if not already present
        if meta.get('ImageRowDir') is None:
            meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

        if meta.get('ImageColumnDir') is None:
            meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

        xml = meta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        logging.debug("Image data has %d elements", image.data.size)

        image.attribute_string = xml
        connection.send_image(image)

    else:
        # 1. Get geometry info from the acquisition header
        # We use the position and orientation of the first acquisition as a base
        base_head = acq.getHead()
        nx, ny, nz = img.shape

        # Calculate slice spacing/thickness
        slice_thickness = fov[-1] / nz * 1000

        for z in range(nz):
            # --- 1. Prepare Data ---
            # Extract the 2D slice [RO, PE] -> [y, x]
            # ISMRMRD from_array(transpose=False) expects [cha, z, y, x]
            # For a single 2D slice, shape is [1, 1, PE, RO]
            slice_data = img[:, :, z].transpose() 
            slice_data = slice_data[np.newaxis, np.newaxis, :, :] 

            image = ismrmrd.Image.from_array(slice_data, transpose=False)

            # --- 2. Update Header ---
            # Start with a header derived from the raw acquisition
            header = mrdhelper.update_img_header_from_raw(image.getHead(), base_head)

            # Update slice-specific parameters
            header.slice = z
            header.image_index = z + 1
            header.image_series_index = 1 

            # Update Position (LPH): Shift the slice position along the slice-normal direction
            # slice_dir is the 'slice selection' or 'normal' vector (read x phase cross product)
            # Most mrdhelpers calculate this, but we ensure it here:
            dir_read = np.array(base_head.read_dir)
            dir_phase = np.array(base_head.phase_dir)
            dir_slice = np.cross(dir_read, dir_phase)

            # Calculate offset for this specific slice relative to the center/start
            # This shifts the center of the volume to the specific slice position
            origin = np.array(base_head.position)
            offset = (z - (nz - 1) / 2.0) * slice_thickness
            slice_position = origin + offset * dir_slice

            header.position = (
                ctypes.c_float(slice_position[0]),
                ctypes.c_float(slice_position[1]),
                ctypes.c_float(slice_position[2])
            )

            # Set Field of View for the slice (Z is now the thickness of one slice)
            image.field_of_view = (
                ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.x),
                ctypes.c_float(mrdHeader.encoding[0].reconSpace.fieldOfView_mm.y),
                ctypes.c_float(slice_thickness)
            )

            image.setHead(header)

            # --- 3. Meta Attributes ---
            meta = ismrmrd.Meta()
            meta['DataRole'] = 'Image'
            meta['ImageProcessingHistory'] = ['OPENRECON', 'PYTHON']
            meta['Keep_image_geometry'] = 1

            # Ensure orientation is in Meta for compatibility
            meta['ImageRowDir'] = ["{:.18f}".format(base_head.read_dir[0]), 
                                "{:.18f}".format(base_head.read_dir[1]), 
                                "{:.18f}".format(base_head.read_dir[2])]
            meta['ImageColumnDir'] = ["{:.18f}".format(base_head.phase_dir[0]), 
                                    "{:.18f}".format(base_head.phase_dir[1]), 
                                    "{:.18f}".format(base_head.phase_dir[2])]

            image.attribute_string = meta.serialize()

            # --- 4. Send ---
            logging.info(f"Sending slice {z}/{nz} at position {slice_position}")
            connection.send_image(image)

    logging.info("Reconstruction complete, closing connection.") 
    connection.send_close()
