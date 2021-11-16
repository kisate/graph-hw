#include "data.hpp"
#include <map>
#include <tuple>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

std::tuple<int, int, int> parse_ind(std::string& ind)
{
	int ind0 = std::stoi(ind.substr(0, ind.find("/")));
	ind.erase(0, ind.find("/") + 1);
	int ind1 = std::stoi(ind.substr(0, ind.find("/")));
	ind.erase(0, ind.find("/") + 1);
	int ind2 = std::stoi(ind.substr(0, ind.find("/")));

	return {ind0, ind1, ind2};
}


RenderObject::RenderObject(GLuint vbo, std::vector<std::vector<int>> indices, std::vector<int> material_indices) {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glGenBuffers(1, &ebo);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);


	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) sizeof(glm::vec3));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) (sizeof(glm::vec3) * 2));

	this->indices = indices;
	this->material_indices = material_indices;
}

void RenderObject::render(std::vector<GLuint>& textures, std::vector<GLuint>& normal_maps)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	
	for (int i = 0; i < indices.size(); ++i)
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices[i].size() * sizeof(indices[i][0]), indices[i].data(), GL_STATIC_DRAW);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textures[material_indices[i]]);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, normal_maps[material_indices[i]]);

		glDrawElements(GL_TRIANGLES, indices[i].size(), GL_UNSIGNED_INT, 0);
	}
	
}

void RenderScene::render()
{
	for (auto& ro : render_objects)
	{
		ro.render(textures, normal_maps);
	}
}


RenderScene::RenderScene(std::string obj_file_path, std::string material_directory_path)
{
	std::map<std::tuple<int, int, int>, int> indices_map;

	tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objmaterials;
	std::string err;

    
    bool success = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, nullptr, &err,
        obj_file_path.c_str(),
        material_directory_path.c_str(),
        true);

    //boilerplate error handling
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }
    if (!success) {
        exit(1);
    }

	for (auto& shape: shapes)
	{
		for (auto& ind: shape.mesh.indices)
		{
			indices_map.emplace(std::tuple(ind.vertex_index, ind.normal_index, ind.texcoord_index), 0);
		}
	}


	for (auto &p: indices_map)
	{
		p.second = vertices.size();
		auto [ind0, ind1, ind2] = p.first;
		vertex v = {
			{attrib.vertices[3*ind0] / 2000.0f, attrib.vertices[3*ind0 + 1] / 2000.0f, attrib.vertices[3*ind0 + 2] / 2000.0f},
			{attrib.normals[3*ind1], attrib.normals[3*ind1 + 1], attrib.normals[3*ind1 + 2]},
			{attrib.texcoords[2*ind2], attrib.texcoords[2*ind2 + 1]}
		};
		vertices.push_back(v);
	}

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

	stbi_set_flip_vertically_on_load(1);

	for (auto& material: objmaterials)
	{
		std::string texture_path = PRACTICE_SOURCE_DIRECTORY "/sponza/" + material.ambient_texname;

		int w;
		int h;
		int comp;
		unsigned char* image = stbi_load(texture_path.c_str(), &w, &h, &comp, STBI_rgb);

		GLuint texture;

		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		glGenerateMipmap(GL_TEXTURE_2D);

		stbi_image_free(image);

		textures.push_back(texture);

		texture_path = PRACTICE_SOURCE_DIRECTORY "/sponza/" + material.normal_texname;

		unsigned char* image2 = stbi_load(texture_path.c_str(), &w, &h, &comp, STBI_rgb);

		GLuint normal_map;

		glGenTextures(1, &normal_map);
		glBindTexture(GL_TEXTURE_2D, normal_map);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, image2);
		glGenerateMipmap(GL_TEXTURE_2D);

		stbi_image_free(image2);

		normal_maps.push_back(normal_map);
	}

	for (auto& shape: shapes)
	{
		std::vector<std::vector<int>> indices;
		std::vector<int> material_indices;
		std::vector<int> cur_indices;

		int last_material_ind = shape.mesh.material_ids[0];

		for (int i = 0; i < shape.mesh.indices.size(); ++i)
		{
			auto index = shape.mesh.indices[i];
			std::tuple<int, int, int> ind = {index.vertex_index, index.normal_index, index.texcoord_index};
			if (shape.mesh.material_ids[i / 3] != last_material_ind)
			{
				material_indices.push_back(last_material_ind);
				indices.push_back(cur_indices);
				cur_indices.clear();
				last_material_ind = shape.mesh.material_ids[i / 3];
			}
			cur_indices.push_back(indices_map[ind]);
		}
		material_indices.push_back(last_material_ind);
		indices.push_back(cur_indices);
		render_objects.emplace_back(vbo, indices, material_indices);
	}
}

CubeMap::CubeMap(int texture_size)
{
	this->texture_size = texture_size;


	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

	for (int i = 0; i < 6; ++i)
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8, texture_size, texture_size, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	}

	
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, texture_size, texture_size);
	// glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	// glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
	// glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, texture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
}

void CubeMap::render(GLuint projection_location, RenderScene& scene, GLuint view_location, GLuint model_location, glm::vec3 center) 
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	for (int i = 0; i < 6; ++i)
	{
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, texture, 0);
		glViewport(0,0, texture_size, texture_size);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		glm::mat4 view(1.f);
		glm::mat4 model(1.f);
		glm::mat4 projection;

		float near = 0.1f;
		float far = 10.f;
		float camera_distance = 0.5;
		float scale = 0.1f;

		projection = glm::ortho(-scale, scale, -scale, scale, near, far);
		view = glm::rotate(view, glm::pi<float>(), {0.f, 0.f, 1.f});

		if (i == 0) {
			// projection = glm::perspective(glm::pi<float>() / 2.f, 1.0f, near, far);

			view = glm::rotate(view, glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
			view = glm::translate(view, {scale, 0.f, 0.f});
		} else if (i == 1) {
			// projection = glm::ortho(-1.f, 1.f, -1.f, 1.f, near, far);
			// view = glm::translate(view, {0.f, 0.f, -camera_distance});
			// view = glm::rotate(view, -glm::pi<float>() / 2.f, {1.f, 0.f, 0.f});

			// view = glm::rotate(view, glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
			view = glm::rotate(view, -glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
			view = glm::translate(view, {-scale, 0.f, 0.f});
		} else if (i == 2) {
			// view = glm::rotate(view, -glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
			view = glm::rotate(view, -glm::pi<float>() / 2.f, {1.f, 0.f, 0.f});
			view = glm::rotate(view, glm::pi<float>(), {0.f, 1.f, 0.f});
			view = glm::translate(view, {0.f, scale, 0.f});
		} else if (i == 3) {
			// view = glm::rotate(view, glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
			view = glm::rotate(view, glm::pi<float>() / 2.f, {1.f, 0.f, 0.f});
			view = glm::rotate(view, glm::pi<float>(), {0.f, 1.f, 0.f});
			view = glm::translate(view, {0.f, -scale, 0.f});
		} else if (i == 4) {
			// view = glm::rotate(view, -glm::pi<float>() / 2.f, {0.f, 0.f, 1.f});
			view = glm::rotate(view, glm::pi<float>(), {0.f, 1.f, 0.f});
			view = glm::translate(view, {0.f, 0.f, scale});

		} else if (i == 5) {
			view = glm::translate(view, {0.f, 0.f, -scale});
		}
		view = glm::translate(view, -center);

		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));;
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));

		scene.render();
	}
}


std::pair<std::vector<vertex>, std::vector<std::uint32_t>> load_obj(std::istream & input)
{
	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;

	for (std::string line; std::getline(input, line);)
	{
		std::istringstream line_stream(line);

		char type;
		line_stream >> type;

		if (type == '#')
			continue;

		if (type == 'v')
		{
			vertex v;
			line_stream >> v.position.x >> v.position.y >> v.position.z;
			vertices.push_back(v);
			continue;
		}

		if (type == 'f')
		{
			std::uint32_t i0, i1, i2;
			line_stream >> i0 >> i1 >> i2;
			--i0;
			--i1;
			--i2;
			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);
			continue;
		}

		throw std::runtime_error("Unknown OBJ row type: " + std::string(1, type));
	}

	return {vertices, indices};
}


void fill_normals(std::vector<vertex> & vertices, std::vector<std::uint32_t> const & indices)
{
	for (auto & v : vertices)
		v.normal = glm::vec3(0.f);

	for (std::size_t i = 0; i < indices.size(); i += 3)
	{
		auto & v0 = vertices[indices[i + 0]];
		auto & v1 = vertices[indices[i + 1]];
		auto & v2 = vertices[indices[i + 2]];

		glm::vec3 n = glm::cross(v1.position - v0.position, v2.position - v0.position);
		v0.normal += n;
		v1.normal += n;
		v2.normal += n;
	}

	for (auto & v : vertices)
		v.normal = glm::normalize(v.normal);
}


Mirror::Mirror(int texture_size): cubemap(texture_size)
{
	static glm::vec3 cube_vertices[] = 
	{
		// -X
		{-1.f, -1.f, -1.f},
		{-1.f, -1.f,  1.f},
		{-1.f,  1.f, -1.f},
		{-1.f,  1.f,  1.f},
		// +X
		{ 1.f, -1.f,  1.f},
		{ 1.f, -1.f, -1.f},
		{ 1.f,  1.f,  1.f},
		{ 1.f,  1.f, -1.f},
		// -Y
		{-1.f, -1.f, -1.f},
		{ 1.f, -1.f, -1.f},
		{-1.f, -1.f,  1.f},
		{ 1.f, -1.f,  1.f},
		// +Y
		{-1.f,  1.f,  1.f},
		{ 1.f,  1.f,  1.f},
		{-1.f,  1.f, -1.f},
		{ 1.f,  1.f, -1.f},
		// -Z
		{ 1.f, -1.f, -1.f},
		{-1.f, -1.f, -1.f},
		{ 1.f,  1.f, -1.f},
		{-1.f,  1.f, -1.f},
		// +Z
		{-1.f, -1.f,  1.f},
		{ 1.f, -1.f,  1.f},
		{-1.f,  1.f,  1.f},
		{ 1.f,  1.f,  1.f},
	};

	static std::uint32_t cube_indices[] = 
	{
		// -X
		0, 1, 2, 2, 1, 3,
		// +X
		4, 5, 6, 6, 5, 7,
		// -Y
		8, 9, 10, 10, 9, 11,
		// +Y
		12, 13, 14, 14, 13, 15,
		// -Z
		16, 17, 18, 18, 17, 19,
		// +Z
		20, 21, 22, 22, 21, 23,
	};
	

	// for (int i = 0; i < 36; ++i)
	// {
	// 	indices.push_back(cube_indices[i]);
	// }

	// for (int i = 0; i < 24; ++i)
	// {
	// 	vertex v = {cube_vertices[i]};
	// 	v.position /= 10.0f;
	// 	vertices.push_back(v);
	// }

	std::vector<vertex> _vertices;
	std::vector<std::uint32_t> _indices;
	{
		std::ifstream bunny_file(PRACTICE_SOURCE_DIRECTORY "/bunny.obj");
		std::tie(_vertices, _indices) = load_obj(bunny_file);
	}
	vertices = _vertices;
	indices = _indices;
	fill_normals(vertices, indices);


	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glGenBuffers(1, &ebo);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) sizeof(glm::vec3));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) (sizeof(glm::vec3) * 2));

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);
}

void Mirror::render()
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap.texture);

	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
};