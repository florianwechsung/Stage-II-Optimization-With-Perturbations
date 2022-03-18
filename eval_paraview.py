# trace generated using paraview version 5.9.1

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

outdirs = [
]

outdir = outdirs[args.idx]

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
coil_opt_visvtu = XMLUnstructuredGridReader(registrationName='coil_opt_vis.vtu', FileName=[outdir + '/coil_opt_vis.vtu'])
coil_opt_visvtu.PointArrayStatus = ['idx']

# Properties modified on coil_opt_visvtu
coil_opt_visvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
coil_opt_visvtuDisplay = Show(coil_opt_visvtu, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'idx'
idxLUT = GetColorTransferFunction('idx')
idxLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 199.5, 0.865003, 0.865003, 0.865003, 399.0, 0.705882, 0.0156863, 0.14902]
idxLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'idx'
idxPWF = GetOpacityTransferFunction('idx')
idxPWF.Points = [0.0, 0.0, 0.5, 0.0, 399.0, 1.0, 0.5, 0.0]
idxPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
coil_opt_visvtuDisplay.Representation = 'Surface'
coil_opt_visvtuDisplay.ColorArrayName = ['POINTS', 'idx']
coil_opt_visvtuDisplay.LookupTable = idxLUT
coil_opt_visvtuDisplay.SelectTCoordArray = 'None'
coil_opt_visvtuDisplay.SelectNormalArray = 'None'
coil_opt_visvtuDisplay.SelectTangentArray = 'None'
coil_opt_visvtuDisplay.OSPRayScaleArray = 'idx'
coil_opt_visvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
coil_opt_visvtuDisplay.SelectOrientationVectors = 'None'
coil_opt_visvtuDisplay.ScaleFactor = 0.3805515493311068
coil_opt_visvtuDisplay.SelectScaleArray = 'idx'
coil_opt_visvtuDisplay.GlyphType = 'Arrow'
coil_opt_visvtuDisplay.GlyphTableIndexArray = 'idx'
coil_opt_visvtuDisplay.GaussianRadius = 0.01902757746655534
coil_opt_visvtuDisplay.SetScaleArray = ['POINTS', 'idx']
coil_opt_visvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
coil_opt_visvtuDisplay.OpacityArray = ['POINTS', 'idx']
coil_opt_visvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
coil_opt_visvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
coil_opt_visvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
coil_opt_visvtuDisplay.ScalarOpacityFunction = idxPWF
coil_opt_visvtuDisplay.ScalarOpacityUnitDistance = 2.094708692126495
coil_opt_visvtuDisplay.OpacityArrayName = ['POINTS', 'idx']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
coil_opt_visvtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 15.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
coil_opt_visvtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 15.0, 1.0, 0.5, 0.0]

# show color bar/color legend
coil_opt_visvtuDisplay.SetScalarBarVisibility(renderView1, True)

# find source
threshold1 = FindSource('Threshold1')

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold2 = Threshold(registrationName='Threshold2', Input=coil_opt_visvtu)
threshold2.Scalars = ['POINTS', 'idx']
threshold2.ThresholdRange = [0.0, 15.0]

# show data in view
threshold2Display = Show(threshold2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold2Display.Representation = 'Surface'
threshold2Display.ColorArrayName = ['POINTS', 'idx']
threshold2Display.LookupTable = idxLUT
threshold2Display.SelectTCoordArray = 'None'
threshold2Display.SelectNormalArray = 'None'
threshold2Display.SelectTangentArray = 'None'
threshold2Display.OSPRayScaleArray = 'idx'
threshold2Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold2Display.SelectOrientationVectors = 'None'
threshold2Display.ScaleFactor = 0.3805515493311068
threshold2Display.SelectScaleArray = 'idx'
threshold2Display.GlyphType = 'Arrow'
threshold2Display.GlyphTableIndexArray = 'idx'
threshold2Display.GaussianRadius = 0.01902757746655534
threshold2Display.SetScaleArray = ['POINTS', 'idx']
threshold2Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold2Display.OpacityArray = ['POINTS', 'idx']
threshold2Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold2Display.DataAxesGrid = 'GridAxesRepresentation'
threshold2Display.PolarAxes = 'PolarAxesRepresentation'
threshold2Display.ScalarOpacityFunction = idxPWF
threshold2Display.ScalarOpacityUnitDistance = 2.094708692126495
threshold2Display.OpacityArrayName = ['POINTS', 'idx']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 15.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 15.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(coil_opt_visvtu, renderView1)

# show color bar/color legend
threshold2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on threshold2
threshold2.ThresholdRange = [0.0, 3.0]

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on threshold2Display
threshold2Display.RenderLinesAsTubes = 1

# Properties modified on threshold2Display
threshold2Display.LineWidth = 10.0

# turn off scalar coloring
ColorBy(threshold2Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(idxLUT, renderView1)

# change solid color
threshold2Display.AmbientColor = [1.0, 0.3333333333333333, 0.0]
threshold2Display.DiffuseColor = [1.0, 0.3333333333333333, 0.0]

# create a new 'XML Structured Grid Reader'
surf_opt_visvts_1 = XMLStructuredGridReader(registrationName='surf_opt_vis.vts', FileName=[outdir + '/surf_opt_vis.vts'])
surf_opt_visvts_1.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B_N/|B|']

# Properties modified on surf_opt_visvts_1
surf_opt_visvts_1.TimeArray = 'None'

# show data in view
surf_opt_visvts_1Display = Show(surf_opt_visvts_1, renderView1, 'StructuredGridRepresentation')

# get color transfer function/color map for 'B_NB'
b_NBLUT = GetColorTransferFunction('B_NB')
b_NBLUT.RGBPoints = [-0.0033363509981390017, 0.231373, 0.298039, 0.752941, 6.078311286057767e-06, 0.865003, 0.865003, 0.865003, 0.003348507620711117, 0.705882, 0.0156863, 0.14902]
b_NBLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'B_NB'
b_NBPWF = GetOpacityTransferFunction('B_NB')
b_NBPWF.Points = [-0.0033363509981390017, 0.0, 0.5, 0.0, 0.003348507620711117, 1.0, 0.5, 0.0]
b_NBPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
surf_opt_visvts_1Display.Representation = 'Surface'
surf_opt_visvts_1Display.ColorArrayName = ['POINTS', 'B_N/|B|']
surf_opt_visvts_1Display.LookupTable = b_NBLUT
surf_opt_visvts_1Display.SelectTCoordArray = 'None'
surf_opt_visvts_1Display.SelectNormalArray = 'None'
surf_opt_visvts_1Display.SelectTangentArray = 'None'
surf_opt_visvts_1Display.OSPRayScaleArray = 'B_N/|B|'
surf_opt_visvts_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
surf_opt_visvts_1Display.SelectOrientationVectors = 'dphi x dtheta'
surf_opt_visvts_1Display.ScaleFactor = 0.12853714188591983
surf_opt_visvts_1Display.SelectScaleArray = 'B_N/|B|'
surf_opt_visvts_1Display.GlyphType = 'Arrow'
surf_opt_visvts_1Display.GlyphTableIndexArray = 'B_N/|B|'
surf_opt_visvts_1Display.GaussianRadius = 0.006426857094295991
surf_opt_visvts_1Display.SetScaleArray = ['POINTS', 'B_N/|B|']
surf_opt_visvts_1Display.ScaleTransferFunction = 'PiecewiseFunction'
surf_opt_visvts_1Display.OpacityArray = ['POINTS', 'B_N/|B|']
surf_opt_visvts_1Display.OpacityTransferFunction = 'PiecewiseFunction'
surf_opt_visvts_1Display.DataAxesGrid = 'GridAxesRepresentation'
surf_opt_visvts_1Display.PolarAxes = 'PolarAxesRepresentation'
surf_opt_visvts_1Display.ScalarOpacityFunction = b_NBPWF
surf_opt_visvts_1Display.ScalarOpacityUnitDistance = 0.11542678172550845

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
surf_opt_visvts_1Display.ScaleTransferFunction.Points = [-0.0033362743647850994, 0.0, 0.5, 0.0, 0.003348507620711117, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
surf_opt_visvts_1Display.OpacityTransferFunction.Points = [-0.0033362743647850994, 0.0, 0.5, 0.0, 0.003348507620711117, 1.0, 0.5, 0.0]

# show color bar/color legend
surf_opt_visvts_1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on surf_opt_visvts_1Display
surf_opt_visvts_1Display.Opacity = 0.7

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
b_NBLUT.ApplyPreset('BuRd', True)

# Rescale transfer function
b_NBLUT.RescaleTransferFunction(-0.001, 0.001)


# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# get the material library
materialLibrary1 = GetMaterialLibrary()

# get color legend/bar for b_NBLUT in view renderView1
b_NBLUTColorBar = GetScalarBar(b_NBLUT, renderView1)
b_NBLUTColorBar.AutoOrient = 0
b_NBLUTColorBar.WindowLocation = 'UpperRightCorner'
b_NBLUTColorBar.Position = [0.9047546634225467, 0.036384976525821594]
b_NBLUTColorBar.Title = 'B_N/|B|'
b_NBLUTColorBar.ComponentTitle = ''
b_NBLUTColorBar.HorizontalTitle = 1
b_NBLUTColorBar.TitleFontSize = 24
b_NBLUTColorBar.LabelFontSize = 24
b_NBLUTColorBar.ScalarBarLength = 0.937981220657277

# Properties modified on b_NBLUTColorBar
b_NBLUTColorBar.UseCustomLabels = 1
# Properties modified on b_NBLUTColorBar
b_NBLUTColorBar.LabelFontSize = 120
b_NBLUTColorBar.CustomLabels = [-0.001, 0.0, 0.001]

# Properties modified on b_NBLUTColorBar
b_NBLUTColorBar.TitleFontSize = 1

# change scalar bar placement
b_NBLUTColorBar.WindowLocation = 'AnyLocation'
b_NBLUTColorBar.Position = [0.8153584672435105, 0.018779342723004737]
b_NBLUTColorBar.ScalarBarLength = 0.9379812206572775

# change scalar bar placement
b_NBLUTColorBar.Position = [0.8502327575922534, 0.022300469483568112]
b_NBLUTColorBar.ScalarBarLength = 0.9379812206572774

# change scalar bar placement
b_NBLUTColorBar.Position = [0.8745636578355623, 0.023474178403755923]
b_NBLUTColorBar.ScalarBarLength = 0.9379812206572774

# change scalar bar placement
b_NBLUTColorBar.Position = [0.95, 0.023474178403755923]
b_NBLUTColorBar.TextPosition = 'Ticks left/bottom, annotations right/top'
b_NBLUTColorBar.LabelColor=[0.0,0.0,0.0]


# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(2466, 1704)

# current camera placement for renderView1
# renderView1.CameraPosition = [1.2*c for c in [3.540096773384972, 2.670701218519498, 1.069240055158276]]
# renderView1.CameraFocalPoint = [0.8257861075674299, 0.7364987079641312, 0.1912662533406153]
# renderView1.CameraViewUp = [-0.020338394502462193, -0.3894407201306181, 0.9208269518281966]
# renderView1.CameraViewUp = [0., -0.4, 0.9]
# renderView1.CameraParallelScale = 1.5803405598160016
# renderView1.CameraParallelScale = 1.3

renderView1.CameraPosition = [0.75*c for c in [4.135115259588702, 5.884633599328879, -0.015191333452321692]]
renderView1.CameraFocalPoint = [0.88017672232078, 0.719054351863155, 0.061585350430194606]
renderView1.CameraViewUp = [0., 0., 1.]
renderView1.CameraParallelScale = 4.


# save screenshot
SaveScreenshot("pv/" + outdir.replace("/", "_").replace(".", "p") + 'pv.png', renderView1, ImageResolution=[2466, 1704],
    TransparentBackground=1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2466, 1704)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [3.540096773384972, 2.670701218519498, 1.069240055158276]
renderView1.CameraFocalPoint = [0.8257861075674299, 0.7364987079641312, 0.1912662533406153]
renderView1.CameraViewUp = [-0.020338394502462193, -0.3894407201306181, 0.9208269518281966]
renderView1.CameraParallelScale = 1.5803405598160016

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
