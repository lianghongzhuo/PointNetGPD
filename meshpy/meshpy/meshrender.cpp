#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <boost/numpy.hpp>
#include <iostream>

#include "GL/osmesa.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#define GLAPIENTRY

// global rending constants
float near = 0.05f;
float far = 1e2f;
float scale = (0x0001) << 0;

// offsets for reading material buffers
int mat_ambient_off = 3;
int mat_diffuse_off = mat_ambient_off + 4;
int mat_specular_off = mat_diffuse_off + 4;
int mat_shininess_off = mat_specular_off + 4;

// offsets for reading lighting buffers
int light_ambient_off = 0;
int light_diffuse_off = light_ambient_off + 4;
int light_specular_off = light_diffuse_off + 4;
int light_position_off = light_specular_off + 4;
int light_direction_off = light_position_off + 3;
int light_spot_cutoff_off = light_direction_off + 3;

void uint2uchar(unsigned int in, unsigned char* out){
  out[0] = (in & 0x00ff0000) >> 16;
  out[1] = (in & 0x0000ff00) >> 8;
  out[2] =  in & 0x000000ff;
}

boost::python::tuple render_mesh(boost::python::list proj_matrices,
                                 unsigned int im_height,
                                 unsigned int im_width,
                                 boost::python::numeric::array verts,
                                 boost::python::numeric::array tris,
                                 boost::python::numeric::array norms,
                                 boost::python::numeric::array mat_props,
                                 boost::python::numeric::array light_props,
				 bool enable_lighting = false,
                                 bool debug = false)
{
  // init rendering vars
  OSMesaContext ctx;
  boost::python::list color_ims;
  boost::python::list depth_ims;
  void *buffer;
  unsigned char* color_result = NULL;
  float* depth_result = NULL;

  // parse input data
  int num_projections = boost::python::len(proj_matrices);
  long int verts_buflen;
  long int tris_buflen;
  long int norms_buflen;
  long int mat_props_buflen;
  long int light_props_buflen;
  void const *verts_raw_buffer;
  void const *tris_raw_buffer;
  void const *norms_raw_buffer;
  void const *mat_props_raw_buffer;
  void const *light_props_raw_buffer;

  // read numpy buffers
  bool verts_readbuf_success = !PyObject_AsReadBuffer(verts.ptr(), &verts_raw_buffer, &verts_buflen);
  bool tris_readbuf_success = !PyObject_AsReadBuffer(tris.ptr(), &tris_raw_buffer, &tris_buflen);
  bool norms_readbuf_success = !PyObject_AsReadBuffer(norms.ptr(), &norms_raw_buffer, &norms_buflen);
  bool mat_props_readbuf_success = !PyObject_AsReadBuffer(mat_props.ptr(), &mat_props_raw_buffer, &mat_props_buflen);
  bool light_props_readbuf_success = !PyObject_AsReadBuffer(light_props.ptr(), &light_props_raw_buffer, &light_props_buflen);

  // cast numpy buffers to C arrays
  const double* verts_buffer = reinterpret_cast<const double*>(verts_raw_buffer);
  const unsigned int* tris_buffer = reinterpret_cast<const unsigned int*>(tris_raw_buffer);
  const double* norms_buffer = reinterpret_cast<const double*>(norms_raw_buffer);
  const double* mat_props_buffer = reinterpret_cast<const double*>(mat_props_raw_buffer);
  const double* light_props_buffer = reinterpret_cast<const double*>(light_props_raw_buffer);

  // read color
  double final_matrix[16];
  unsigned char colorBytes[3];
  colorBytes[0] = (unsigned char)mat_props_buffer[0];
  colorBytes[1] = (unsigned char)mat_props_buffer[1];
  colorBytes[2] = (unsigned char)mat_props_buffer[2];

  // compute num vertices
  unsigned int num_verts = verts_buflen / (3 * sizeof(double));
  unsigned int num_tris = tris_buflen / (3 * sizeof(unsigned int));
  unsigned int num_norms = norms_buflen / (3 * sizeof(double));
  if (debug) {
    std::cout << "Num vertices " << num_verts << std::endl;
    std::cout << "Num tris " << num_tris << std::endl;
    std::cout << "Num norms " << num_norms << std::endl;
    std::cout << "Color " << (int)colorBytes[0] << " " << (int)colorBytes[1] << " " << (int)colorBytes[2] << std::endl;
  }

  // create an RGBA-mode context
  ctx = OSMesaCreateContextExt( OSMESA_RGBA, 16, 0, 0, NULL );
  if (!ctx) {
    printf("OSMesaCreateContext failed!\n");
  }

  // allocate the image buffer
  buffer = malloc( im_width * im_height * 4 * sizeof(GLubyte) );
  if (!buffer) {
    printf("Alloc image buffer failed!\n");
  }

  // bind the buffer to the context and make it current
  if (!OSMesaMakeCurrent( ctx, buffer, GL_UNSIGNED_BYTE, im_width, im_height )) {
    printf("OSMesaMakeCurrent failed!\n");
  }
  OSMesaPixelStore(OSMESA_Y_UP, 0);

  // setup material properties
  if (enable_lighting) {
    GLfloat mat_ambient[4];
    GLfloat mat_diffuse[4];
    GLfloat mat_specular[4];
    GLfloat mat_shininess[1];
    mat_ambient[0] = (GLfloat)mat_props_buffer[mat_ambient_off + 0];
    mat_ambient[1] = (GLfloat)mat_props_buffer[mat_ambient_off + 1];
    mat_ambient[2] = (GLfloat)mat_props_buffer[mat_ambient_off + 2];
    mat_ambient[3] = (GLfloat)mat_props_buffer[mat_ambient_off + 3];

    mat_diffuse[0] = (GLfloat)mat_props_buffer[mat_diffuse_off + 0];
    mat_diffuse[1] = (GLfloat)mat_props_buffer[mat_diffuse_off + 1];
    mat_diffuse[2] = (GLfloat)mat_props_buffer[mat_diffuse_off + 2];
    mat_diffuse[3] = (GLfloat)mat_props_buffer[mat_diffuse_off + 3];

    mat_specular[0] = (GLfloat)mat_props_buffer[mat_specular_off + 0];
    mat_specular[1] = (GLfloat)mat_props_buffer[mat_specular_off + 1];
    mat_specular[2] = (GLfloat)mat_props_buffer[mat_specular_off + 2];
    mat_specular[3] = (GLfloat)mat_props_buffer[mat_specular_off + 3];

    mat_shininess[0] = (GLfloat)mat_props_buffer[mat_shininess_off + 0];

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_SMOOTH);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

    // setup lighting properties
    GLfloat light_ambient[4];
    GLfloat light_diffuse[4];
    GLfloat light_specular[4];
    GLfloat light_position[4];
    GLfloat light_direction[3];
    GLfloat light_spot_cutoff[1];

    light_ambient[0] = (GLfloat)light_props_buffer[light_ambient_off + 0];
    light_ambient[1] = (GLfloat)light_props_buffer[light_ambient_off + 1];
    light_ambient[2] = (GLfloat)light_props_buffer[light_ambient_off + 2];
    light_ambient[3] = (GLfloat)light_props_buffer[light_ambient_off + 3];

    light_diffuse[0] = (GLfloat)light_props_buffer[light_diffuse_off + 0];
    light_diffuse[1] = (GLfloat)light_props_buffer[light_diffuse_off + 1];
    light_diffuse[2] = (GLfloat)light_props_buffer[light_diffuse_off + 2];
    light_diffuse[3] = (GLfloat)light_props_buffer[light_diffuse_off + 3];

    light_specular[0] = (GLfloat)light_props_buffer[light_specular_off + 0];
    light_specular[1] = (GLfloat)light_props_buffer[light_specular_off + 1];
    light_specular[2] = (GLfloat)light_props_buffer[light_specular_off + 2];
    light_specular[3] = (GLfloat)light_props_buffer[light_specular_off + 3];

    light_position[0] = (GLfloat)light_props_buffer[light_position_off + 0];
    light_position[1] = (GLfloat)light_props_buffer[light_position_off + 1];
    light_position[2] = (GLfloat)light_props_buffer[light_position_off + 2];
    light_position[3] = 1.0; // always set w to 1

    light_direction[0] = (GLfloat)light_props_buffer[light_direction_off + 0];
    light_direction[1] = (GLfloat)light_props_buffer[light_direction_off + 1];
    light_direction[2] = (GLfloat)light_props_buffer[light_direction_off + 2];

    light_spot_cutoff[0] = (GLfloat)light_props_buffer[light_spot_cutoff_off + 0];

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light_direction);
    glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, light_spot_cutoff);

    if (debug) {
      std::cout << "Light pos " << light_position[0] << " " << light_position[1] << " " << light_position[2] << " " << light_position[3] << std::endl;
      std::cout << "Light dir " << light_direction[0] << " " << light_direction[1] << " " << light_direction[2] << std::endl;
    }

    // enable lighting
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
  }

  // set color
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);

  // setup rendering
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  for (unsigned int k = 0; k < num_projections; k++) {
    // load next projection matrix
    boost::python::object proj_matrix_obj(proj_matrices[k]);
    long int proj_buflen;
    void const *proj_raw_buffer;
    bool proj_readbuf_success = !PyObject_AsReadBuffer(proj_matrix_obj.ptr(),
                                                       &proj_raw_buffer,
                                                       &proj_buflen);
    const double* projection = reinterpret_cast<const double*>(proj_raw_buffer);
    if (debug) {
      std::cout << "Proj Matrix " << k << std::endl;
      std::cout << projection[0] << " " << projection[1] << " " << projection[2] << " " << projection[3] << std::endl;
      std::cout << projection[4] << " " << projection[5] << " " << projection[6] << " " << projection[7] << std::endl;
      std::cout << projection[8] << " " << projection[9] << " " << projection[10] << " " << projection[11] << std::endl;
    }

    // create projection
    double inv_width_scale  = 1.0 / (im_width * scale);
    double inv_height_scale = 1.0 / (im_height * scale);
    double inv_width_scale_1 = inv_width_scale - 1.0;
    double inv_height_scale_1_s = -(inv_height_scale - 1.0);
    double inv_width_scale_2 = inv_width_scale * 2.0;
    double inv_height_scale_2_s = -inv_height_scale * 2.0;
    double far_a_near = far + near;
    double far_s_near = far - near;
    double far_d_near = far_a_near / far_s_near;
    final_matrix[ 0] = projection[0+2*4] * inv_width_scale_1 + projection[0+0*4] * inv_width_scale_2;
    final_matrix[ 4] = projection[1+2*4] * inv_width_scale_1 + projection[1+0*4] * inv_width_scale_2;
    final_matrix[ 8] = projection[2+2*4] * inv_width_scale_1 + projection[2+0*4] * inv_width_scale_2;
    final_matrix[ 12] = projection[3+2*4] * inv_width_scale_1 + projection[3+0*4] * inv_width_scale_2;

    final_matrix[ 1] = projection[0+2*4] * inv_height_scale_1_s + projection[0+1*4] * inv_height_scale_2_s;
    final_matrix[ 5] = projection[1+2*4] * inv_height_scale_1_s + projection[1+1*4] * inv_height_scale_2_s;
    final_matrix[ 9] = projection[2+2*4] * inv_height_scale_1_s + projection[2+1*4] * inv_height_scale_2_s;
    final_matrix[13] = projection[3+2*4] * inv_height_scale_1_s + projection[3+1*4] * inv_height_scale_2_s;

    final_matrix[ 2] = projection[0+2*4] * far_d_near;
    final_matrix[ 6] = projection[1+2*4] * far_d_near;
    final_matrix[10] = projection[2+2*4] * far_d_near;
    final_matrix[14] = projection[3+2*4] * far_d_near - (2*far*near)/far_s_near;

    final_matrix[ 3] = projection[0+2*4];
    final_matrix[ 7] = projection[1+2*4];
    final_matrix[11] = projection[2+2*4];
    final_matrix[15] = projection[3+2*4];

    // load projection and modelview matrices
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(final_matrix);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // render mesh
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, im_width, im_height);
    for (unsigned int i = 0; i < num_tris; ++i) {
      glColor3ubv(colorBytes);
      glBegin(GL_POLYGON);

      unsigned int a = tris_buffer[3*i + 0];
      unsigned int b = tris_buffer[3*i + 1];
      unsigned int c = tris_buffer[3*i + 2];

      glNormal3dv(&norms_buffer[3 * a]);
      glVertex3dv(&verts_buffer[3 * a]);
      glNormal3dv(&norms_buffer[3 * b]);
      glVertex3dv(&verts_buffer[3 * b]);
      glNormal3dv(&norms_buffer[3 * c]);
      glVertex3dv(&verts_buffer[3 * c]);
      glEnd();
    }

    glFinish();

    // pull color buffer and flip y axis
    int i, j;
    GLint out_width, out_height, bytes_per_depth, color_type;
    GLboolean succeeded;
    unsigned char* p_color_buffer;
    succeeded = OSMesaGetColorBuffer(ctx, &out_width, &out_height, &color_type, (void**)&p_color_buffer);
    if (color_result == NULL)
      color_result = new unsigned char[3 * out_width * out_height];
    for (i = 0; i < out_width; i++) {
      for (j = 0; j < out_height; j++) {
        int di = i + j * out_width; // index in color buffer
        int ri = i + j * out_width; // index in rendered image
        color_result[3*ri+0] = p_color_buffer[4*di+0];
        color_result[3*ri+1] = p_color_buffer[4*di+1];
        color_result[3*ri+2] = p_color_buffer[4*di+2];
      }
    }

    // pull depth buffer and flip y axis
    unsigned short* p_depth_buffer;
    succeeded = OSMesaGetDepthBuffer(ctx, &out_width, &out_height, &bytes_per_depth, (void**)&p_depth_buffer);
    if (depth_result == NULL)
      depth_result = new float[out_width * out_height];
    for(i = 0; i < out_width; i++){
      for(j = 0; j < out_height; j++){
        int di = i + j * out_width; // index in depth buffer
        int ri = i + (out_height-1-j)*out_width; // index in rendered image
        if (p_depth_buffer[di] == USHRT_MAX) {
          depth_result[ri] = 0.0f;
        }
        else {
          depth_result[ri] = near / (1.0f - ((float)p_depth_buffer[di] / USHRT_MAX));
        }
      }
    }

    // append ndarray color image to list
    boost::python::tuple color_shape = boost::python::make_tuple(im_height, im_width, 3);
    boost::numpy::dtype color_dt = boost::numpy::dtype::get_builtin<unsigned char>();
    boost::numpy::ndarray color_arr = boost::numpy::from_data(color_result, color_dt, color_shape,
                                                              boost::python::make_tuple(color_shape[1]*color_shape[2]*sizeof(unsigned char),
                                                                                        color_shape[2]*sizeof(unsigned char),
                                                                                        sizeof(unsigned char)),
                                                              boost::python::object());
    color_ims.append(color_arr.copy());

    // append ndarray depth image to list
    boost::python::tuple depth_shape = boost::python::make_tuple(im_height, im_width);
    boost::numpy::dtype depth_dt = boost::numpy::dtype::get_builtin<float>();
    boost::numpy::ndarray depth_arr = boost::numpy::from_data(depth_result, depth_dt, depth_shape,
                                                              boost::python::make_tuple(depth_shape[1]*sizeof(float),
                                                                                        sizeof(float)),
                                                              boost::python::object());
    depth_ims.append(depth_arr.copy());
  }

  // free the image buffer
  free( buffer );

  // destroy the context
  OSMesaDestroyContext( ctx );

  //return depth_ims;
  boost::python::tuple ret_tuple = boost::python::make_tuple(color_ims, depth_ims);

  if (color_result != NULL)
    delete [] color_result;
  if (depth_result != NULL)
    delete [] depth_result;

  return ret_tuple;
}

// Test function for multiplying an array by a scalar
boost::python::list mul_array(boost::python::numeric::array data, int x)
{
  // Access a built-in type (an array)
  boost::python::numeric::array a = data;
  long int bufLen;
  void const *buffer;
  bool isReadBuffer = !PyObject_AsReadBuffer(a.ptr(), &buffer, &bufLen);
  std::cout << "BUFLEN " << bufLen << std::endl;
  const double* test = reinterpret_cast<const double*>(buffer);
  int s = bufLen / sizeof(double);
  double* mult = new double[s];
  for (int i = 0; i < s; i++) {
    mult[i] = x * test[i];
  }

  const boost::python::tuple& shape = boost::python::extract<boost::python::tuple>(a.attr("shape"));
  std::cout << "Shape " << boost::python::extract<int>(shape[0]) << " " << boost::python::extract<int>(shape[1]) << std::endl;
  boost::numpy::dtype dt = boost::numpy::dtype::get_builtin<double>();
  boost::numpy::ndarray result = boost::numpy::from_data(mult, dt, shape,
                                                         boost::python::make_tuple(shape[0]*sizeof(double),
                                                                                   sizeof(double)),
                                                         boost::python::object());

  boost::python::list l;
  l.append(result);
  return l;
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(meshrender) {
  Py_Initialize();
  boost::numpy::initialize();
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

  def("mul_array", &mul_array);
  def("render_mesh", &render_mesh);
}
