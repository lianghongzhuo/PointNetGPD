from unittest import TestCase
import numpy as np
from meshpy import Mesh3D

class TestMesh(TestCase):

    def test_init(self):
        verts = [[1,0,0],[0,1,0],[-1,0,0],[0,0,1]]
        tris = [[3,0,1],[3,1,2],[3,2,0],[0,2,1]]
        d = 1.2
        m = Mesh3D(verts, tris, density=d)
        self.assertTrue(isinstance(m, Mesh3D))
        self.assertEqual(m.vertices.shape, (4,3))
        self.assertEqual(m.vertices.tolist(), verts)
        self.assertEqual(m.triangles.shape, (4,3))
        self.assertEqual(m.triangles.tolist(), tris)
        self.assertEqual(m.density, d)
        self.assertEqual(m.bb_center.tolist(), [0.0, 0.5, 0.5])
        self.assertEqual(m.centroid.tolist(), [0.0, 0.25, 0.25])

    def test_read(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        self.assertTrue(isinstance(m, Mesh3D))
        self.assertTrue(m.vertices.shape == (4,3))
        self.assertTrue(m.triangles.shape == (4,3))

    def test_min_coords(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        self.assertTrue(m.min_coords().tolist() == [-1.0, 0.0, 0.0])

    def test_max_coords(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        self.assertTrue(m.max_coords().tolist() == [1.0, 1.0, 1.0])

    def test_bounding_box(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        minc, maxc = m.bounding_box()
        self.assertTrue(minc.tolist() == [-1.0, 0.0, 0.0])
        self.assertTrue(maxc.tolist() == [1.0, 1.0, 1.0])

    def test_bounding_box_mesh(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        bbm = m.bounding_box_mesh()
        self.assertTrue(isinstance(bbm, Mesh3D))
        self.assertEqual(bbm.vertices.shape, (8,3))
        self.assertEqual(bbm.triangles.shape, (12,3))
        self.assertEqual(bbm.bb_center.tolist(), [0.0, 0.5, 0.5])
        self.assertEqual(bbm.centroid.tolist(), [0.0, 0.5, 0.5])

    def test_principal_dims(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        pd = m.principal_dims()
        self.assertEqual(pd.tolist(), [2.0, 1.0, 1.0])

    def test_support(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        s = m.support(np.array([1,0,0]))
        self.assertEqual(s.shape, (3,))

    def test_tri_centers(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        centers = m.tri_centers()
        self.assertEqual(centers.shape, (4,3))
        self.assertTrue([0.0, 1.0/3.0, 0.0] in centers.tolist())

    def test_tri_normals(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        n = m.tri_normals(True)
        self.assertEqual(n.shape, (4,3))
        self.assertTrue([0.0, -1.0, 0.0] in n.tolist())
        self.assertTrue([0.0, 0.0, -1.0] in n.tolist())

    def test_total_volume(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        v = m.total_volume()
        self.assertEqual(v, 1.0/3.0)

    def test_covariance(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        cv = m.covariance()
        actual_cov = np.array([[1.0/30.0, 0.0, 0.0],
                      [0.0, 1.0/30.0, 1.0/60.0],
                      [0.0, 1.0/60.0, 1.0/30.0]])
        self.assertEqual(np.round(cv, 5).tolist(), np.round(actual_cov, 5).tolist())

    def test_remove_bad_tris(self):
        m = Mesh3D.load('test/data/bad_tetrahedron.obj', 'test/cache')
        self.assertEqual(m.triangles.shape[0], 6)
        m.remove_bad_tris()
        self.assertEqual(m.triangles.shape[0], 4)

    def test_remove_unreferenced_vertices(self):
        m = Mesh3D.load('test/data/bad_tetrahedron.obj', 'test/cache')
        self.assertEqual(m.vertices.shape[0], 6)
        m.remove_bad_tris()
        m.remove_unreferenced_vertices()
        self.assertEqual(m.vertices.shape[0], 4)

    def test_center_vertices_avg(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m.center_vertices_avg()
        self.assertEqual(m.centroid.tolist(), [0,0,0])
        self.assertTrue([0,-0.25,0.75] in m.vertices.tolist())

    def test_center_vertices_bb(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m.center_vertices_bb()
        self.assertEqual(m.bb_center.tolist(), [0,0,0])
        self.assertTrue([0,-0.5,0.5] in m.vertices.tolist())

    def test_normalize_vertices(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m.normalize_vertices()
        new_verts = [[-0.3536, 0, -1],
                     [0.3536, 0.7071, 0],
                     [-0.3536, 0, 1],
                     [0.3536, -0.7071, 0]]
        self.assertEqual(np.round(m.bb_center, 5).tolist(), [0,0,0])
        self.assertEqual(np.round(m.vertices, 4).tolist(), new_verts)

    def test_copy(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        x = m.copy()
        self.assertEqual(m.vertices.tolist(), x.vertices.tolist())
        self.assertEqual(m.triangles.tolist(), x.triangles.tolist())

    def test_subdivide(self):
        pass
        #m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        #x = m.subdivide()
        #self.assertEqual(m.vertices.shape[0], 10)
        #self.assertEqual(m.triangles.shape[0], 16)

    def test_transform(self):
        pass

    def test_get_T_surface_obj(self):
        pass

    def test_rescale_dimension(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m2 = m.copy()
        m2.rescale_dimension(0.5, Mesh3D.ScalingTypeMin)
        self.assertEqual(m2.min_coords().tolist(), [-0.5, 0.0, 0.0])
        self.assertEqual(m2.max_coords().tolist(), [0.5, 0.5, 0.5])
        m3 = m.copy()
        m3.rescale_dimension(0.5, Mesh3D.ScalingTypeMin)
        self.assertEqual(m3.min_coords().tolist(), [-0.5, 0.0, 0.0])
        self.assertEqual(m3.max_coords().tolist(), [0.5, 0.5, 0.5])

    def test_rescale(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m.rescale(0.5)
        self.assertEqual(m.min_coords().tolist(), [-0.5, 0.0, 0.0])
        self.assertEqual(m.max_coords().tolist(), [0.5, 0.5, 0.5])

    def test_convex_hull(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        cvh = m.convex_hull()
        self.assertEqual(cvh.min_coords().tolist(), [-1, 0, 0])
        self.assertEqual(cvh.max_coords().tolist(), [1, 1, 1])

    def test_stable_poses(self):
        m = Mesh3D.load('test/data/tetrahedron.obj', 'test/cache')
        m.center_of_mass = m.centroid
        stps = m.stable_poses()
        self.assertEqual(len(stps), 4)

    def test_visualize(self):
        pass

if __name__ == '__main__':
    unittest.main()
