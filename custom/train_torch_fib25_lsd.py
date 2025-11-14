import json
import logging
import math
import numpy as np
import os
import sys
import torch

from funlib.learn.torch.models import UNet
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

# --- LSD-SPECIFIC IMPORT ---
# We assume this package is available, as it was in train.py
from gp import AddLocalShapeDescriptor

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

# --- DATA PLACEHOLDER ---
# As requested, we'll keep the data loading section as a placeholder.
# The format (Zarr) and structure are assumed to be the same.
data_dir = 'funke/fib25/training/'
samples = [ # Placeholder
    'trvol-250-1.zarr',
    'trvol-250-2.zarr',
    'tstvol-520-1.zarr',
    'tstvol-520-2.zarr',
]
# needs to match order of samples (small to large)
probabilities = [0.05, 0.05, 0.45, 0.45]

# --- MODEL DEFINITION ---
# This class is copied directly from train_pytorch.py.
# It perfectly matches the logic of a U-Net followed by a 
# 1x1x1 conv + sigmoid, as seen in mknet.py.
class Convolve(torch.nn.Module):

    def __init__(
            self,
            model,
            in_channels,
            out_channels,
            kernel_size=(1,1,1)):

        super().__init__()

        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        conv = torch.nn.Conv3d

        self.conv_pass = torch.nn.Sequential(
                            conv(
                                self.in_channels,
                                self.out_channels,
                                self.kernel_size),
                            torch.nn.Sigmoid())

    def forward(self, x):

        y = self.model.forward(x)

        return self.conv_pass(y)

# --- CUSTOM LOSS FUNCTION ---
# This is new. We need a weighted MSE loss to match the
# logic from mknet.py:
# tf.losses.mean_squared_error(gt_embedding, embedding, loss_weights_embedding)
class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # We want the un-reduced loss to apply weights
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, prediction, target, weights):
        # Calculate element-wise loss
        loss = self.mse(prediction, target)
        
        # Apply weights
        weighted_loss = loss * weights

        # Return the mean of the weighted loss
        return weighted_loss.mean()

# --- CUSTOM BATCH FILTER ---
# This class is copied directly from train.py.
# It's needed to unmask the background voxels.
class UnmaskBackground(BatchFilter):
    ''' 
    We want to mask out losses for LSDs at the boundary
    between neurons while not simultaneously masking out
    losses for LSDs at raw=0. Therefore we should add
    (1 - background mask) to gt_embedding_scale after we add
    the LSDs in the AddLocalShapeDescriptor node.
    '''
    def __init__(self, target_mask, background_mask):
        self.target_mask = target_mask
        self.background_mask = background_mask
    def process(self, batch, request):
        batch[self.target_mask].data = np.logical_or(
                batch[self.target_mask].data,
                np.logical_not(batch[self.background_mask].data))

# This is the custom node to create labels_mask by copying labels
class Copy(BatchFilter):
    '''A simple gunpowder node to copy an array.
    
    Args:
        array_in (:class:`ArrayKey`): The array to copy from.
        array_out (:class:`ArrayKey`): The array to copy to.
    '''
    def __init__(self, array_in, array_out):
        self.array_in = array_in
        self.array_out = array_out

    def setup(self):
        self.provides(self.array_out, self.spec[self.array_in].copy())
    
    # --- THIS IS THE CORRECTED PREPARE METHOD ---
    def prepare(self, request):
        
        # 1. Start with the downstream request
        deps = request.copy()

        # 2. Check if the array we provide (labels_mask) is requested
        if self.array_out in deps:
            
            # 3. Add our dependency (labels)
            array_in_spec = deps[self.array_out].copy() # Get the ROI from labels_mask
            if self.array_in in deps:
                # Merge ROIs if 'labels' is already requested by another node
                deps[self.array_in].roi = deps[self.array_in].roi.union(
                    array_in_spec.roi
                )
            else:
                # Add 'labels' to the dependency request
                deps[self.array_in] = array_in_spec
            
            # --- THIS IS THE FIX ---
            # 4. Remove the key we *provide* (labels_mask)
            #    from the *upstream* request (deps).
            del deps[self.array_out]
            # --- END OF FIX ---
            
        return deps
    # --- END OF CORRECTION ---

    # --- THIS PROCESS METHOD IS STILL CORRECT ---
    def process(self, batch, request):
        
        # If the downstream node requested 'array_out' (labels_mask)...
        if self.array_out in request:
            
            # 'batch[self.array_in]' (labels) is guaranteed to exist
            # from our 'prepare' method. We crop it to the
            # size requested for 'array_out' (labels_mask).
            
            requested_roi = request[self.array_out].roi
            batch[self.array_out] = batch[self.array_in].crop(requested_roi)
    # --- END OF CORRECTION ---

def train_until(max_iteration):

    # --- MODEL PARAMETERS (from mknet.py) ---
    in_channels = 1
    num_fmaps = 12
    fmap_inc_factors = 6 # 5 or 4
    downsample_factors = [(2,2,2),(2,2,2),(3,3,3)]
    out_channels = 10 # 10 LSDs

    # --- MODEL INSTANTIATION ---
    unet = UNet(
            in_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors)                                           #given inputs, builds all layers for us
    
    # Use the Convolve wrapper to add the final 1x1x1 conv + sigmoid
    model = Convolve(
            unet,
            in_channels=num_fmaps,  # U-Net output fmaps
            out_channels=out_channels) # Final LSD fmaps                  #adds final 1x1x1 conv layer
    
    print(unet)
    print(unet.modules)
    
    # --- LOSS AND OPTIMIZER (from mknet.py) ---
    loss = WeightedMSELoss()
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4,
            betas=(0.95, 0.999),
            eps=1e-8) # Explicitly set epsilon to match TF                #optimizer matches tensorflow ver.

    # --- SHAPES (from mknet.py) ---
    # Input shape for 'train_net'
    input_shape = Coordinate((196, 196, 196)) #from 196
    
    # Output shape is determined by the U-Net architecture
    # Given the input_shape and downsample_factors, the output
    # shape from the original affinity example (84, 84, 84) is correct.
    output_shape = Coordinate((84, 84, 84))

    voxel_size = Coordinate((8,)*3)                                     #'Coordinate' part of gunpowder, tells gunpowder 'a single voxel = 8x8x8 units in physical world'

    # --- ARRAY KEYS (for LSDs) ---
    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    embedding = ArrayKey('PREDICTED_EMBEDDING')
    gt_embedding = ArrayKey('GT_EMBEDDING')
    gt_embedding_scale = ArrayKey('GT_EMBEDDING_SCALE')
    embedding_gradient = ArrayKey('EMBEDDING_GRADIENT')

    
    input_size = input_shape * voxel_size                               # When you multiply two Coordinate objects, it performs element-wise multiplication
    output_size = output_shape * voxel_size                             # Thus these define 'real-world size' of input & output
    
    #max labels padding calculated (copied from train.py)
    #labels_padding = Coordinate((608,768,768))  #was 608,768,768
    labels_padding = Coordinate((128, 128, 128))  #was 608,768,768

    request = BatchRequest()                                            # When requesting to gunpowder, it doesn't request in terms of voxels, but in terms of physical size
    request.add(raw, input_size)                                        # Hence if data has different resolution, can automatically give appropriate number of voxels
    request.add(labels, output_size) 
    request.add(labels_mask, output_size)                               # In BatchRequest is stored, the exact 'amounts' of each 'key' to be requested
    request.add(gt_embedding, output_size)                              # request raw of input sizes // labels of output sizes // ...
    request.add(gt_embedding_scale, output_size)

    snapshot_request = BatchRequest({
        embedding: request[gt_embedding],                               # Additional, separate request for different timing (something to do with these being built mid-training)
        embedding_gradient: request[gt_embedding]
    })

    # --- DATA SOURCE (Placeholder) ---
    data_sources = tuple(
            # ZarrSource(
            #         os.path.join(data_dir, sample),                     # Each "Sample" is a zarr file (like a zip file), in which exists multiple datasets
            #         {                                                   # Inside each sample are individual datasets of raw, neuron_ids, mask etc
            #             raw: 'volumes/raw',
            #             labels: 'volumes/labels/neuron_ids',
            #             #labels_mask: 'volumes/labels/mask',
            #         },
            #         {
            #             raw: ArraySpec(interpolatable=True),            # rules for how to handle each ArrayKey, interpolatable = is 'continuous', so if needed, can 'average'
            #             labels: ArraySpec(interpolatable=False),
            #             #labels_mask: ArraySpec(interpolatable=False)
            #         }                                                   # A + B means "take the data from node A, and pass it to node B" --> creates a pipeline
            #     ) +
            ZarrSource(
                os.path.join(data_dir, sample),
                {
                    raw: 'volumes/raw',
                    labels: 'volumes/labels/neuron_ids',
                },
                {
                    raw: ArraySpec(interpolatable=True,  voxel_size=voxel_size, dtype=np.float32),
                    labels: ArraySpec(interpolatable=False, voxel_size=voxel_size, dtype=np.uint64),
                }
            )
            #Copy(labels, labels_mask) + #Added
            #Normalize(raw) +
            #Pad(raw, 0) + 
            #Pad(labels, 0) +
            #Pad(labels_mask, 0) +
            #RandomLocation() +
            #Reject(mask=labels_mask, min_masked=0.5)                ##############################################################################################
        
            #Pad(raw, None) +
            #Pad(labels, None) + #was Pad(labels, labels_padding)
            #Pad(labels_mask, None) + #was Pad(labels_mask, labels_padding)
            #RandomLocation() + #was just RandomLocation(min_masked=0.5, mask=labels_mask)
            #Reject(mask=labels_mask, min_masked=0.5) #did not exist
        
            Copy(labels, labels_mask) +
            Normalize(raw) +
            # Pad(raw, None) +
            # Pad(labels, labels_padding) + 
            # Pad(labels_mask, labels_padding) +
            finite_pad = Coordinate((128,128,128))
            Pad(raw, finite_pad)
            Pad(labels, finite_pad)
            Pad(labels_mask, finite_pad)
            RandomLocation(min_masked=0.5, mask=labels_mask)
            
            for sample in samples
        )

    # --- GUNPOWDER PIPELINE ---
    train_pipeline = data_sources

    train_pipeline += RandomProvider(probabilities=probabilities)       # Based on probabilities (defined above), the batch requested will be 'randomly' from the four .zarr files
                                                                        # Useful as the first two files are smaller than the last two, ensuring uniform? sampling?
    # Augmentations (copied from train.py)
    train_pipeline += ElasticAugment(
            control_point_spacing=[40, 40, 40],                         # Augmentations to data (in preciesly the order presented)
            jitter_sigma=[0, 0, 0],                                     # Random 90deg rotation
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=8)

    train_pipeline += SimpleAugment()                                   # Random flip & mirror

    train_pipeline += ElasticAugment(                                   # Non-linear warping, 90deg rotation, small shift (some microscopy data specific error stuff)
            control_point_spacing=[40,40,40],
            jitter_sigma=[2,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.01,
            prob_shift=0.01,
            max_misalign=1,
            subsample=8)

    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)        # Random intensity changes

    train_pipeline += GrowBoundary(labels, labels_mask, steps=1)        # Grows boundary 'between neurons' by 1 voxel (and modifies labels_mask accordingly)

    # --- LSD-SPECIFIC NODES (from train.py) ---
    # Replaces AddAffinities and BalanceLabels
    train_pipeline += AddLocalShapeDescriptor(                          # Creates ground truth - gt_embedding, gt_embedding_scale
            labels,
            gt_embedding,
            lsds_mask=gt_embedding_scale,
            sigma=80,
            downsample=2)

    # Add the unmasking node
    train_pipeline += UnmaskBackground(gt_embedding_scale, labels_mask)  # labels_mask has 0 for background (default) and 0 for boundary (GrowBoundary)
    # --- END LSD-SPECIFIC NODES ---                                     # We want only the boundaries to be 0, so that we can learn only the boundaries
                                                                         # As such, we 'add back' the background
    train_pipeline += IntensityScaleShift(raw, 2,-1) #intensity stuff

    # PyTorch-specific nodes for channel dimensions
    # train_pipeline += Unsqueeze([raw])
    # train_pipeline += Unsqueeze([gt_embedding, gt_embedding_scale])      # Should... handle the dimension requirement diff between pytorch and tensorflow
    # ensure dtype
    train_pipeline += EnsureDtype(raw, np.float32)
    train_pipeline += EnsureDtype(labels, np.uint64)

    # add a channel dim as first spatial axis (C)
    train_pipeline += Unsqueeze([raw], dims=[0])                 # raw: (C=1,Z,Y,X)
    train_pipeline += Unsqueeze([gt_embedding, gt_embedding_scale], dims=[0])  # (C=10,Z,Y,X) and (C=10,Z,Y,X)

    train_pipeline += PreCache(                                          # Buffer for slow cpu stuff above, to be loaded into fast gpu stuff below
            cache_size=4,
            num_workers=1)

    # --- PYTORCH TRAIN NODE (Modified for LSDs) ---
    train_pipeline += Train(                                             # Performs training...
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'x': raw
            },
            loss_inputs={
                # These must match the order in WeightedMSELoss.forward()
                0: embedding,          # (prediction) from model output 0
                1: gt_embedding,       # (target) from batch
                2: gt_embedding_scale  # (weights) from batch
            },
            outputs={
                0: embedding
            },
            gradients={
                0: embedding_gradient
            },
            save_every=10000, # Copied from train.py, iterations
            log_dir='log')

    # Squeeze channel dimensions back out
    train_pipeline += Squeeze([raw, gt_embedding, gt_embedding_scale, embedding])   # Cleanup - reverts changes (unsqueeze, intensity shift), for better visualization/analysis

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    # --- SNAPSHOT NODE (Modified for LSDs) ---
    train_pipeline += Snapshot({                                               # Separate Node just for batch saving
                raw: 'volumes/raw',                                            # Note, only embedding, and embedding_gradient is 'additionally requested'
                labels: 'volumes/labels/neuron_ids',
                gt_embedding: 'volumes/gt_embedding',
                embedding: 'volumes/pred_embedding',
                labels_mask: 'volumes/labels/mask',
                embedding_gradient: 'volumes/embedding_gradient'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_embedding: np.float32
            },
            every=1000, # Copied from train.py
            output_filename='batch_{iteration}.zarr', # Changed to zarr
            additional_request=snapshot_request)

    train_pipeline += PrintProfilingStats(every=10)                          # Every 10 iter, do runtime analysis

    print("Starting LSD training with PyTorch...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)                                         # Constructs the actual pipeline and STARTS the process (workers working, batch stuff)
    print("Training finished")

if __name__ == '__main__':

    iterations = 1000 # Placeholder
    train_until(iterations)

