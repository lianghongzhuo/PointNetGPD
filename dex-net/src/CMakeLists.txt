
PYTHON_ADD_MODULE(meshrender dexnet/meshrender.cpp)
#FILE(COPY render_mesh.py DESTINATION .)
INSTALL(TARGETS meshrender DESTINATION dexnet)