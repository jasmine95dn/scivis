import numpy as np
# importing vtk is required for correctly setting up object factories
import vtk
# vtk numpy helper
import vtkmodules.numpy_interface.dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
# base class for Python algorithms
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import (
    vtkCommand,
    vtkFloatArray,
    vtkLookupTable,
    vtkPoints,
)
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkImageData,
    vtkPolyData,
    VTK_POLY_LINE,
)
from vtkmodules.vtkFiltersCore import (
    vtkGlyph3D,
    vtkPolyDataNormals,
    vtkTubeFilter,
)
from vtkmodules.vtkFiltersGeneral import (
    vtkWarpScalar,
)
from vtkmodules.vtkFiltersGeometry import (
    vtkStructuredGridGeometryFilter,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTexture,
)


class BilinearInterpolator(object):
    """ Bilinear interpolation on a uniform grid. """

    def __init__(self, dimensions, origin, spacing, data, fill_value=np.nan):
        self._dimensions = np.asarray(dimensions)
        self._grid_min = np.asarray(origin)
        self._spacing = np.asarray(spacing)
        self._grid_max = self._grid_min + self._spacing * (self._dimensions - 1)
        self._fill_value = fill_value
        self._data = data
        assert(self._data.shape[:2] == tuple(dimensions[:2]))

    def __call__(self, X):
        """ Given a set of points as the rows of X, compute the bilinear interpolation at each point.
            For consistency, if X has only one dimension, the output will have shape (1, ...).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # transform to grid coordinates
        X_grid = (X - self._grid_min) / self._spacing
        # get cell index
        cell_index = np.floor(X_grid).astype(np.int)
        # clip index boundaries
        cell_index[:, 0] = np.clip(cell_index[:, 0], 0, self._dimensions[0] - 2)
        cell_index[:, 1] = np.clip(cell_index[:, 1], 0, self._dimensions[1] - 2)
        # cell local coordinates
        X_local = X_grid - cell_index
        alpha, beta = X_local[:, 0], X_local[:, 1]
        # scalars at cell nodes
        s00 = self._data[cell_index[:, 0] + 0, cell_index[:, 1] + 0]
        s10 = self._data[cell_index[:, 0] + 1, cell_index[:, 1] + 0]
        s01 = self._data[cell_index[:, 0] + 0, cell_index[:, 1] + 1]
        s11 = self._data[cell_index[:, 0] + 1, cell_index[:, 1] + 1]
        # bilinear interpolation
        s = (1. - alpha) * (1. - beta) * s00 + \
            alpha * (1. - beta) * s10 + \
            (1. - alpha) * beta * s01 + \
            alpha * beta * s11
        # remove invalid values
        valid = (X[:, 0] >= self._grid_min[0]) & (X[:, 0] <= self._grid_max[0]) & \
                (X[:, 1] >= self._grid_min[1]) & (X[:, 1] <= self._grid_max[1])
        s[~valid] = self._fill_value
        return s


class CentralDifferencesDerivatives(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1,
                                        inputType='vtkImageData', outputType='vtkImageData')

    def GetInputArrayToProcess(self, id, data_object):
        """ emulates the missing vtkAlgorithm::GetInputArrayToProcess method """
        info = self.GetInputArrayInformation(id)
        field = info.Get(vtkDataObject.FIELD_ASSOCIATION())
        name = info.Get(vtkDataObject.FIELD_NAME())
        if field == vtkDataObject.FIELD_ASSOCIATION_POINTS:
            return data_object.PointData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_CELLS:
            return data_object.CellData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_NONE:
            return data_object.FieldData[name]
        else:  # not implemented
            return None

    def RequestData(self, request, inInfo, outInfo):
        # get input and output objects
        input_data = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output_data = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 0))
        if not input_data or not output_data:
            return 0

        # copy input to output
        output_data.ShallowCopy(input_data.VTKObject)

        # get input data
        dimensions = input_data.GetDimensions()[:2]
        spacing = input_data.GetSpacing()[:2]
        input_scalars = self.GetInputArrayToProcess(0, input_data)
        input_scalars = input_scalars.reshape(dimensions, order='F')

        # these arrays hold the output values
        gradient = np.zeros(input_scalars.shape + (2,))
        hessian = np.zeros(input_scalars.shape + (2, 2))

        # the input grid is defined by
        #   dimensions: number of samples in x- and y-directions
        #   spacing:    distance between two grid nodes in x- and y-directions
        #   input_scalars:    2D array holding the scalar data
        #                     the scalar s_ij is given by input_scalars[i, j]
        #
        # output:
        #   gradient: 2D array holding the gradient vectors at grid nodes
        #             gradient[i,j] is the 2D gradient vector
        #   hessian:  2D array holding the Hessian matrices at grid nodes
        #             hessian[i,j] is the 2x2 Hessian matrix

        ## c) calculate gradient and hessian
        
        gradient = np.asarray(np.gradient(input_scalars))
        hessian = np.zeros((2, 2) + input_scalars.shape)
        for k, gradient_k in enumerate(gradient):
            temp_gradient = np.gradient(gradient_k)
            for l, gradient_kl in enumerate(temp_gradient):
                hessian[k,l,:,:] = gradient_kl


        gradient = gradient.reshape((-1, 2), order='F')
        output_data.PointData.append(gradient, 'gradient')
        hessian = hessian.reshape((-1, 4), order='F')
        output_data.PointData.append(hessian, 'hessian')

        return 1


class GradientDescent(VTKPythonAlgorithmBase):
    def __init__(self):
        self._position = np.array([0.1, 0.1])
        self._use_second_derivatives = False
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1,
                                        inputType='vtkImageData', outputType='vtkPolyData')

    def SetPosition(self, x, y):
        self._position = np.array([x, y])
        self.Modified()

    def SetUseSecondDerivatives(self, b):
        self._use_second_derivatives = b
        self.Modified()

    def GetInputArrayToProcess(self, id, data_object):
        """ emulates the missing vtkAlgorithm::GetInputArrayToProcess method """
        info = self.GetInputArrayInformation(id)
        field = info.Get(vtkDataObject.FIELD_ASSOCIATION())
        name = info.Get(vtkDataObject.FIELD_NAME())
        if field == vtkDataObject.FIELD_ASSOCIATION_POINTS:
            return data_object.PointData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_CELLS:
            return data_object.CellData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_NONE:
            return data_object.FieldData[name]
        else:  # not implemented
            return None

    def RequestData(self, request, inInfo, outInfo):
        # get input and output objects
        input_data = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output_data = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        if not input_data or not output_data:
            return 0

        # get input data
        dimensions = input_data.GetDimensions()[:2]
        origin = input_data.GetBounds()[::2][:2]
        spacing = input_data.GetSpacing()[:2]
        grid_max = input_data.GetBounds()[1::2][:2]
        grid_coords = tuple(np.linspace(origin[i], grid_max[i], dimensions[i]) for i in range(2))

        # get input scalars, gradient and Hessian
        scalars_array = self.GetInputArrayToProcess(0, input_data)
        scalars_array = scalars_array.reshape(list(dimensions), order='F')
        gradient_array = self.GetInputArrayToProcess(1, input_data)
        gradient_array = gradient_array.reshape(list(dimensions) + [2], order='F')
        hessian_array = self.GetInputArrayToProcess(2, input_data)
        hessian_array = hessian_array.reshape(list(dimensions) + [2, 2], order='F')

        # use above data arrays to perform bilinear interpolation (you may also use your own code instead)
        scalars = BilinearInterpolator(dimensions, origin, spacing, scalars_array, fill_value=0)
        gradient = BilinearInterpolator(dimensions, origin, spacing, gradient_array, fill_value=0)
        hessian = BilinearInterpolator(dimensions, origin, spacing, hessian_array, fill_value=np.eye(2))

        # starting point
        x = np.copy(self._position)

        # list of 3D points, will connected to a polyline 
        poly_line = []
        # example output, replace this with your own code!
        #poly_line.append(np.r_[x, scalars(x)[0]])
        #poly_line.append(np.r_[x + (.5, .5), scalars(x + (.5, .5))[0]])

        for _ in range(30):
            poly_line.append(x)
            x -= 0.1 * np.gradient(x)





        # create output
        output = dsa.WrapDataObject(vtkPolyData())
        poly_line = np.stack(poly_line, axis=0)
        output.Points = poly_line
        output.Allocate()
        output.InsertNextCell(VTK_POLY_LINE, poly_line.shape[0], list(range(poly_line.shape[0])))
        output.BuildCells()
        output_data.ShallowCopy(output.VTKObject)

        return 1


class CriticalPoints(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1,
                                        inputType='vtkImageData', outputType='vtkPolyData')

    def GetInputArrayToProcess(self, id, data_object):
        """ emulates the missing vtkAlgorithm::GetInputArrayToProcess method """
        info = self.GetInputArrayInformation(id)
        field = info.Get(vtkDataObject.FIELD_ASSOCIATION())
        name = info.Get(vtkDataObject.FIELD_NAME())
        if field == vtkDataObject.FIELD_ASSOCIATION_POINTS:
            return data_object.PointData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_CELLS:
            return data_object.CellData[name]
        elif field == vtkDataObject.FIELD_ASSOCIATION_NONE:
            return data_object.FieldData[name]
        else:  # not implemented
            return None

    def RequestData(self, request, inInfo, outInfo):
        # get input and output objects
        input_data = dsa.WrapDataObject(vtkImageData.GetData(inInfo[0], 0))
        output_data = dsa.WrapDataObject(vtkPolyData.GetData(outInfo, 0))
        if not input_data or not output_data:
            return 0

        # get input data
        dimensions = input_data.GetDimensions()[:2]
        origin = input_data.GetBounds()[::2][:2]
        spacing = input_data.GetSpacing()[:2]
        grid_max = input_data.GetBounds()[1::2][:2]
        grid_coords = tuple(np.linspace(origin[i], grid_max[i], dimensions[i]) for i in range(2))

        # get input scalars, gradient and Hessian
        scalars_array = self.GetInputArrayToProcess(0, input_data)
        scalars_array = scalars_array.reshape(list(dimensions), order='F')
        gradient_array = self.GetInputArrayToProcess(1, input_data)
        gradient_array = gradient_array.reshape(list(dimensions) + [2], order='F')
        hessian_array = self.GetInputArrayToProcess(2, input_data)
        hessian_array = hessian_array.reshape(list(dimensions) + [2, 2], order='F')

        # use above data arrays to perform bilinear interpolation (you may also use your own code instead)
        scalars = BilinearInterpolator(dimensions, origin, spacing, scalars_array, fill_value=0)
        gradient = BilinearInterpolator(dimensions, origin, spacing, gradient_array, fill_value=0)
        hessian = BilinearInterpolator(dimensions, origin, spacing, hessian_array, fill_value=0)

        # output: list of 3D points
        critical_points = []




        # each row represents a color (R, G, B) for each critical point
        color = np.full((len(critical_points), 3), 255, dtype=np.uint8)




        # create output poly data
        output = dsa.WrapDataObject(vtkPolyData())
        if len(critical_points) > 0:
            output.Points = np.stack(critical_points, axis=0)
        # add color array
        output.PointData.append(color, 'color')
        output.Allocate()
        output_data.ShallowCopy(output.VTKObject)

        return 1


def main():
    ### a) 
    ## load the given image
    png_reader = vtkPNGReader()
    png_reader.SetFileName("./interleavedrules.png")

    ## create texture object
    texture = vtkTexture()
    texture.SetInputConnection(png_reader.GetOutputPort())

    # create input scalar field
    image_dimensions = np.array([101, 101], dtype=np.int)
    image_min = np.array([-1.2, -1.2])
    image_max = np.array([1.2, 1.2])
    image_spacing = (image_max - image_min) / (image_dimensions - 1)
    x, y = np.meshgrid(*[np.linspace(image_min[i], image_max[i], image_dimensions[i]) for i in range(2)], indexing='ij')
    scalars = .5 * (np.cos(2. * x * np.pi) * np.cos(1. * y * np.pi) + np.exp(-(x**2 + y**2 + .5 * x * y) / 2.))
    scalars = scalars.reshape(-1, order='F')
    
    image = dsa.WrapDataObject(vtkImageData())
    image.SetDimensions(image_dimensions[0], image_dimensions[1], 1)
    image.SetOrigin(image_min[0], image_min[1], 0.)
    image.SetSpacing(image_spacing[0], image_spacing[1], 1.)
    image.PointData.append(scalars, 'scalars')

    # add texture coordinates
    texture_coords = vtkFloatArray()
    texture_coords.SetName('TextureCoordinates')
    texture_coords.SetNumberOfComponents(2)
    texture_coords.SetNumberOfTuples(scalars.shape[0])
    #image.PointData.SetTCoords(texture_coords)
    
    # map to numpy array
    texture_coords = vtk_to_numpy(texture_coords)
    #print(texture_coords)
    texture_coords[:, 0] = 0.   # u coordinates
    texture_coords[:, 1] =  0. # v coordinates
    #print(texture_coords)
    ## a) linear transform scalar values to the range [0,1]
    texture_coords[:,0] = scalars * (-1/4) + 0.5 ## new_u coordinates
    texture_coords = numpy_to_vtk(texture_coords)
    image.PointData.SetTCoords(texture_coords)
    
    ## b) place a sphere in the scene
    sphere = vtkSphereSource()
    

    # pipeline for filters to be implemented
    derivative_filter = CentralDifferencesDerivatives()
    derivative_filter.SetInputDataObject(image.VTKObject)
    # set scalars to be used for computing derivatives
    derivative_filter.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'scalars')

    gradient_decent = GradientDescent()
    gradient_decent.SetInputConnection(derivative_filter.GetOutputPort())
    # set scalar, gradient, Hessian data array inputs
    gradient_decent.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'scalars')
    gradient_decent.SetInputArrayToProcess(1, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'gradient')
    gradient_decent.SetInputArrayToProcess(2, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'hessian')

    critical_points = CriticalPoints()
    critical_points.SetInputConnection(derivative_filter.GetOutputPort())
    # set scalar, gradient, Hessian data array inputs
    critical_points.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'scalars')
    critical_points.SetInputArrayToProcess(1, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'gradient')
    critical_points.SetInputArrayToProcess(2, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'hessian')

    # create sphere glyphs at each critical point
    glyph_spheres = vtkSphereSource()
    glyph_spheres.SetRadius(.02)
    glyph_spheres.SetThetaResolution(32)
    glyph_spheres.SetPhiResolution(32)
    glyphs = vtkGlyph3D()
    glyphs.SetSourceConnection(glyph_spheres.GetOutputPort())
    glyphs.SetInputConnection(critical_points.GetOutputPort())
    glyphs.ScalingOff()
    # output color value at each input point to the corresponding glyph
    glyphs.SetInputArrayToProcess(3, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'color')
    glyphs.SetColorModeToColorByScalar()
    # create mapper and actor for glyphs
    glyph_mapper = vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
    glyph_actor = vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetSpecular(.2)
    glyph_actor.GetProperty().SetSpecularPower(20)

    # translate each vertex in z-direction depending on scalar value
    warp = vtkWarpScalar()
    warp.SetInputDataObject(image.VTKObject)
    warp.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'scalars')
    # create geometry from structured grid
    geometry_filter = vtkStructuredGridGeometryFilter()
    geometry_filter.SetInputConnection(warp.GetOutputPort())
    # compute normals
    normals_filter = vtkPolyDataNormals()
    normals_filter.SetInputConnection(geometry_filter.GetOutputPort())
    # create mapper and actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(normals_filter.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)

    ## a) set mapper and the defined texture to actor
    #png_mapper = vtkPolyDataMapper()
    #png_actor = vtkActor()
    #png_actor.SetMapper(png_mapper)
    actor.SetTexture(texture)

    # renders a collection of vtkActors
    renderer = vtkRenderer()
    # add actors to be rendered
    renderer.AddActor(actor)
    renderer.AddActor(glyph_actor)
    
    # set background color
    renderer.SetBackground(82/255., 87/255., 110/255.)
    renderer.GetActiveCamera().Elevation(-45.)
    renderer.GetActiveCamera().Dolly(.2)

    # create render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(900, 750)
    # antialiasing: turn this on if your GPU can handle it
    # render_window.SetMultiSamples(64)

    # create interactor, that handles user input
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # set up picking
    picker = vtkPropPicker()
    interactor.SetPicker(picker)

    # called on left mouse button press
    def left_button_press(caller, event):
        position = caller.GetEventPosition()
        if not picker.Pick(position[0], position[1], 0., renderer):
            return
        if picker.GetActor() != actor:
            return
        x, y, z = picker.GetPickPosition()
        # pass picked position to gradient descent filter
        gradient_decent.SetPosition(x, y)

    interactor.AddObserver(vtkCommand.LeftButtonPressEvent, left_button_press, 1.0)

    # set input style
    interactor_style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)


    # run event loop
    interactor.Start()


if __name__ == '__main__':
    main()
