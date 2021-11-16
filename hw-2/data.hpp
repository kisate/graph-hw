#pragma once
#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include <stb_image.h>
#include <tiny_obj_loader.h>

struct vertex
{
	glm::vec3 position;
	glm::vec3 normal;
    glm::vec2 texcoord;
};

struct RenderObject{
	GLuint vao;
	GLuint vbo;
	GLuint ebo;

	std::vector<std::vector<int>> indices;
	std::vector<int> material_indices;

    RenderObject(GLuint vbo, std::vector<std::vector<int>> indices, std::vector<int> material_indices);

	void render(std::vector<GLuint>& textures, std::vector<GLuint>& normal_maps);
};

struct RenderScene{
	std::vector<vertex> vertices;
	std::vector<RenderObject> render_objects;
	std::vector<GLuint> textures;
	std::vector<GLuint> normal_maps;

	RenderScene(std::string obj_file_path, std::string material_directory_path);
	void render();
};

struct CubeMap {
	GLuint texture;
	
	std::vector<GLuint> rbos;
	std::vector<GLuint> fbos;

	GLuint rbo;
	GLuint fbo;

	int texture_size;

	CubeMap(int texture_size);

	void render(GLuint projection_location, RenderScene& scene, GLuint view_location, GLuint model_location, glm::vec3 center); 
};

struct Mirror {
	CubeMap cubemap;

	GLuint vao;
	GLuint vbo;
	GLuint ebo;	

	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;

	glm::vec3 center;

	Mirror(int texture_size);
	void render();
};