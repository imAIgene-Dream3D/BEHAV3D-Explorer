# BEHAV3D napari plugin package
from behav3d.core.opencl_env import silence_pyopencl_compiler_warnings
from behav3d.napari._matplotlib_guard import install_matplotlib_guard
from behav3d.napari._napari_patches import install_napari_patches

# Must run before anything imports pyplot: the analysis backends draw from
# background QThreads, where a Qt Matplotlib backend builds figure windows off
# the GUI thread (constant warning bubbles, and a crash when one is dismissed).
install_matplotlib_guard()

# napari surfaces warnings in its notification/activity area, so install this
# as soon as the plugin loads: the OpenCL device probes on the segmentation
# page can build kernels long before apoc_train (which configures PyOpenCL
# itself) is first imported.
silence_pyopencl_compiler_warnings()
install_napari_patches()
