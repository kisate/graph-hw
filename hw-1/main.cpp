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
#include <random>

std::string to_string(std::string_view str)
{
	return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
	throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
	throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
	R"(#version 330 core

uniform mat4 view;
uniform mat4 transform;
uniform mat4 transform2;
uniform mat4 transform3;

layout (location = 0) in vec2 in_position;
layout (location = 1) in float height;
layout (location = 2) in vec4 in_color;

out vec4 color;

void main()
{
	gl_Position = view * transform * transform2 * transform3 * vec4(in_position, height, 1.0);
	color = in_color;
}
)";

const char fragment_shader_source[] =
	R"(#version 330 core

in vec4 color;

layout (location = 0) out vec4 out_color;

void main()
{
	out_color = color;
}
)";

GLuint create_shader(GLenum type, const char *source)
{
	GLuint result = glCreateShader(type);
	glShaderSource(result, 1, &source, nullptr);
	glCompileShader(result);
	GLint status;
	glGetShaderiv(result, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Shader compilation failed: " + info_log);
	}
	return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
	GLuint result = glCreateProgram();
	glAttachShader(result, vertex_shader);
	glAttachShader(result, fragment_shader);
	glLinkProgram(result);

	GLint status;
	glGetProgramiv(result, GL_LINK_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint info_log_length;
		glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
		std::string info_log(info_log_length, '\0');
		glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
		throw std::runtime_error("Program linkage failed: " + info_log);
	}

	return result;
}

struct vec3
{
	float x;
	float y;
	float z;
};

struct vec2
{
	float x;
	float y;
};

struct metaball
{
	vec2 pos;
	vec2 dir;
	float r;
	float c;
};

struct vertex
{
	vec3 position;
	std::uint8_t color[4];
};

std::vector<metaball> metaballs = {
	{{0.f, 0.f}, {1.f, 0.f}, 1.f, 1.f},
	{{1.f, 0.f}, {-0.5f, -0.5f}, 1.f, 0.9f},
	{{1.f, 0.f}, {0.5f, -0.5f}, 2.f, -1.f},
	{{1.f, 0.f}, {-1.f, 0.5f}, 0.5f, 1.5f}};

float calc_metaball(float x, float y, int i)
{
	float x_i = metaballs[i].pos.x;
	float y_i = metaballs[i].pos.y;
	float c = metaballs[i].c;
	float r = metaballs[i].r;

	return c * exp(-((x - x_i) * (x - x_i) + (y - y_i) * (y - y_i)) / r / r);
}

std::vector<vertex> vertices;
std::vector<uint32_t> indices;

std::vector<float> heights;
std::vector<vertex> isoline_vertices;
std::vector<uint32_t> isoline_indices;
std::map<std::pair<int, int>, uint32_t> interpolated_indices;

vec3 interpolate_coords(int id1, int id2, float c)
{
	vertex vertex1 = vertices[id1];
	vertex vertex2 = vertices[id2];

	float s = abs(vertex1.position.z - c) + abs(vertex2.position.z - c);

	float a1 = abs(vertex1.position.z - c) / s;
	float a2 = abs(vertex2.position.z - c) / s;

	return {
		vertex1.position.x * a2 + vertex2.position.x * a1,
		vertex1.position.y * a2 + vertex2.position.y * a1,
		c + 0.002f};
}

int side_size = 100;
float n_iso = 3;

float range = 4.f;

void add_isoline(float c)
{
	std::map<std::pair<int, int>, uint32_t> interpolated_indices;
	for (int i = 0; i < side_size - 1; ++i)
	{
		for (int j = 0; j < side_size - 1; ++j)
		{
			int corner_ids[4] = {
				i * side_size + j,
				i * side_size + j + 1,
				(i + 1) * side_size + j + 1,
				(i + 1) * side_size + j};

			float corners[4] = {
				heights[corner_ids[0]] - c,
				heights[corner_ids[1]] - c,
				heights[corner_ids[2]] - c,
				heights[corner_ids[3]] - c};

			float field = (corners[0] + corners[1] + corners[2] + corners[3]) / 4;

			int pluses = 0;

			for (int n = 0; n < 4; ++n)
			{
				pluses += corners[n] > 0;
			}

			std::vector<std::pair<int, int>> to_interpolate;

			if (pluses == 4 || pluses == 0)
			{
				continue;
			}

			else if (pluses == 3 || pluses == 1)
			{
				for (int n = 0; n < 4; ++n)
				{
					if (corners[n] * corners[(n + 3) % 4] < 0 && corners[n] * corners[(n + 1) % 4] < 0)
					{
						to_interpolate.emplace_back(n, (n + 3) % 4);
						to_interpolate.emplace_back(n, (n + 1) % 4);
					}
				}
			}

			else if (pluses == 2)
			{
				if (corners[0] * corners[1] > 0)
				{
					to_interpolate.emplace_back(0, 3);
					to_interpolate.emplace_back(1, 2);
				}
				else if (corners[0] * corners[3] > 0)
				{
					to_interpolate.emplace_back(0, 1);
					to_interpolate.emplace_back(3, 2);
				}
				else if (corners[0] * field > 0)
				{
					to_interpolate.emplace_back(0, 1);
					to_interpolate.emplace_back(1, 2);
					to_interpolate.emplace_back(0, 3);
					to_interpolate.emplace_back(3, 2);
				}
				else
				{
					to_interpolate.emplace_back(1, 0);
					to_interpolate.emplace_back(0, 3);
					to_interpolate.emplace_back(1, 2);
					to_interpolate.emplace_back(3, 2);
				}
			}

			for (auto &p : to_interpolate)
			{
				std::pair<int, int> ids = {corner_ids[p.first], corner_ids[p.second]};
				if (interpolated_indices.contains(ids))
				{
					isoline_indices.push_back(interpolated_indices[ids]);
				}
				else
				{
					isoline_vertices.push_back(
						{interpolate_coords(ids.first, ids.second, c),
						 {200, 200, 200, 255}});
					isoline_indices.push_back(isoline_vertices.size() - 1);
					interpolated_indices[ids] = isoline_vertices.size() - 1;
				}
			}
		}
	}
}

void update_iso(float max_z, float min_z)
{

	isoline_vertices.clear();
	isoline_indices.clear();

	for (int i = 0; i <= int(n_iso); ++i)
	{
		float part = 1.f - 0.8f * float(i) / n_iso; 
		add_isoline(max_z * part);
		add_isoline(min_z * part);
	}
}

void update_vertices()
{
	vertices.resize(side_size * side_size);
	heights.resize(side_size * side_size);
	float max_z = -1e9;
	float min_z = 1e9;
	for (int i = 0; i < side_size; ++i)
	{
		for (int j = 0; j < side_size; ++j)
		{
			float x = 2.f * range * i / (side_size - 1) - range;
			float y = 2.f * range * j / (side_size - 1) - range;
			float z = 0;

			for (int i = 0; i < metaballs.size(); ++i)
			{
				z += calc_metaball(x, y, i);
			}

			heights[i * side_size + j] = z / range;

			if (z / range > max_z)
			{
				max_z = z / range;
			}

			if (z / range < min_z)
			{
				min_z = z / range;
			}

			vertices[i * side_size + j] = vertex({{x / range, -y / range, z / range}, {0,0,0, 255}});
		}
	}

	for (int i = 0; i < side_size; ++i)
	{
		for (int j = 0; j < side_size; ++j)
		{
			vec3 position = vertices[i * side_size + j].position;
			float part = (position.z - min_z) / (max_z - min_z);

			vertices[i * side_size + j] = vertex({position, {int(255*part),0,int(255*part),255}});
		}
	}

	update_iso(max_z, min_z);
}

void update_indices()
{
	indices.clear();
	for (int i = 0; i < side_size - 1; ++i)
	{
		for (int j = 0; j < side_size - 1; ++j)
		{
			indices.push_back(i * side_size + j + 1);
			indices.push_back(i * side_size + j + side_size);
			indices.push_back(i * side_size + j);
			indices.push_back(i * side_size + j + 1);
			indices.push_back(i * side_size + j + 1 + side_size);
			indices.push_back(i * side_size + j + side_size);
		}
	}
}

void update_metaballs(float dt)
{
	for (auto &ball : metaballs)
	{
		ball.pos.x += dt * ball.dir.x;
		ball.pos.y += dt * ball.dir.y;
		if (abs(ball.pos.x) > range)
		{
			ball.dir.x *= -1;
		}
		if (abs(ball.pos.y) > range)
		{
			ball.dir.y *= -1;
		}
	}
}


void send_vertices_vector(std::vector<vertex> &vertices) {
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), 0, GL_STATIC_DRAW);

	std::vector<vec2> coords;
	std::vector<float> heights;
	std::vector<std::tuple< uint8_t, uint8_t,uint8_t,uint8_t>> colors;

	for (auto v: vertices)
	{
		vec2 c = {v.position.x, v.position.y};
		coords.push_back(c);
		heights.push_back(v.position.z);
		colors.emplace_back(v.color[0], v.color[1], v.color[2], v.color[3]);
	}

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void *)0);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void *) (sizeof(float) * 2 * coords.size()));
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4*sizeof(uint8_t), (void *) (sizeof(float) * 3 * coords.size()));

	glBufferSubData(GL_ARRAY_BUFFER, 0, (sizeof(float) * 2 * coords.size()), coords.data());
	glBufferSubData(GL_ARRAY_BUFFER, (sizeof(float) * 2 * coords.size()), (sizeof(float) * 1 * coords.size()), heights.data());
	glBufferSubData(GL_ARRAY_BUFFER, (sizeof(float) * 3 * coords.size()), (sizeof(colors[0])  * coords.size()), colors.data());
}

void send_vertices(GLuint vbo, GLuint ebo, GLuint vao, std::vector<vertex> vertices)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindVertexArray(vao);

	std::vector<std::tuple< uint8_t, uint8_t,uint8_t,uint8_t>> colors;
	std::vector<float> heights;
	for (auto v: vertices)
	{
		colors.emplace_back(v.color[0], v.color[1], v.color[2], v.color[3]);
		heights.push_back(v.position.z);
	}

	glBufferSubData(GL_ARRAY_BUFFER, (sizeof(float) * 2 * vertices.size()), (sizeof(float) * 1 * vertices.size()), heights.data());
	glBufferSubData(GL_ARRAY_BUFFER, (sizeof(float) * 3 * vertices.size()), (sizeof(colors[0])  * colors.size()), colors.data());
}

void send_indices(GLuint vbo, GLuint ebo, GLuint vao, std::vector<vertex> &vertices, std::vector<uint32_t> &indices)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindVertexArray(vao);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), 0, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void *)0);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void *) (sizeof(float) * 2 * vertices.size()));
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4*sizeof(uint8_t), (void *) (sizeof(float) * 3 * vertices.size()));

	std::vector<vec2> coords;
	for (auto v : vertices) 
	{
		vec2 c = {v.position.x, v.position.y};
		coords.push_back(c);
	}

	glBufferSubData(GL_ARRAY_BUFFER, 0, (sizeof(float) * 2 * vertices.size()), coords.data());
}

void send_iso(GLuint vbo, GLuint ebo, GLuint vao)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindVertexArray(vao);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, isoline_indices.size()*sizeof(isoline_indices[0]), isoline_indices.data(), GL_STATIC_DRAW);
	
	send_vertices_vector(isoline_vertices);
}

std::vector<vertex> axes = {
	{{-1.f, 0.f, 0.f}, {0, 0, 0, 0}},
	{{1.f, 0.f, 0.f}, {0, 0, 0, 0}},
	{{0.f, -1.f, 0.f}, {0, 0, 0, 0}},
	{{0.f, 1.f, 0.f}, {0, 0, 0, 0}},
	{{0.f, 0.f, -0.5f}, {0, 0, 0, 0}},
	{{0.f, 0.f, 0.5f}, {0, 0, 0, 0}}};

int main()
try
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
		sdl2_fail("SDL_Init: ");

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window *window = SDL_CreateWindow("Graphics course practice 4",
										  SDL_WINDOWPOS_CENTERED,
										  SDL_WINDOWPOS_CENTERED,
										  800, 600,
										  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

	if (!window)
		sdl2_fail("SDL_CreateWindow: ");

	int width, height;
	SDL_GetWindowSize(window, &width, &height);

	SDL_GLContext gl_context = SDL_GL_CreateContext(window);
	if (!gl_context)
		sdl2_fail("SDL_GL_CreateContext: ");

	if (auto result = glewInit(); result != GLEW_NO_ERROR)
		glew_fail("glewInit: ", result);

	if (!GLEW_VERSION_3_3)
		throw std::runtime_error("OpenGL 3.3 is not supported");


	std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-2.f, 2.f);
	std::uniform_real_distribution<> dis2(0.5f, 1.5f);
    
	for (int i = 0; i < 50; ++i)
	{
		metaball m = {
			{dis(gen), dis(gen)},
			{dis(gen), dis(gen)},
			dis2(gen),
			dis(gen)*0.5f
		};
		metaballs.push_back(
			m
		);
	}

	glClearColor(0.8f, 0.8f, 1.f, 0.f);

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint transform_location = glGetUniformLocation(program, "transform");
	GLuint transform2_location = glGetUniformLocation(program, "transform2");
	GLuint transform3_location = glGetUniformLocation(program, "transform3");

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;

	std::map<SDL_Keycode, bool> button_down;

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	GLuint ebo;
	glGenBuffers(1, &ebo);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	update_indices();
	update_vertices();
	send_indices(vbo, ebo, vao, vertices, indices);

	GLuint ax_vbo;
	glGenBuffers(1, &ax_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, ax_vbo);

	GLuint ax_vao;
	glGenVertexArrays(1, &ax_vao);
	glBindVertexArray(ax_vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	send_vertices_vector(axes);

	GLuint iso_vbo;
	glGenBuffers(1, &iso_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, iso_vbo);

	GLuint iso_vao;
	glGenVertexArrays(1, &iso_vao);
	glBindVertexArray(iso_vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	GLuint iso_ebo;
	glGenBuffers(1, &iso_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iso_ebo);

	float scale = 1.1;
	float speed = 1;

	glEnable(GL_DEPTH_TEST);
	// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	float angle1 = -0.5f;
	float angle2 = -1.8f;
	float angle3 = 0;

	bool running = true;
	while (running)
	{
		for (SDL_Event event; SDL_PollEvent(&event);)
			switch (event.type)
			{
			case SDL_QUIT:
				running = false;
				break;
			case SDL_WINDOWEVENT:
				switch (event.window.event)
				{
				case SDL_WINDOWEVENT_RESIZED:
					width = event.window.data1;
					height = event.window.data2;
					glViewport(0, 0, width, height);
					break;
				}
				break;
			case SDL_KEYDOWN:
				button_down[event.key.keysym.sym] = true;
				break;
			case SDL_KEYUP:
				button_down[event.key.keysym.sym] = false;
				break;
			}

		if (!running)
			break;

		auto now = std::chrono::high_resolution_clock::now();
		float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
		last_frame_start = now;
		time += dt;

		if (button_down[SDLK_LEFT])
		{
			angle1 -= speed * dt;
		}
		if (button_down[SDLK_RIGHT])
		{
			angle1 += speed * dt;
		}
		if (button_down[SDLK_UP])
		{
			angle2 += speed * dt;
		}
		if (button_down[SDLK_DOWN])
		{
			angle2 -= speed * dt;
		}
		if (button_down[SDLK_a])
		{
			angle3 += speed * dt;
		}
		if (button_down[SDLK_d])
		{
			angle3 -= speed * dt;
		}
		if (button_down[SDLK_o])
		{
			n_iso += dt;
		}
		if (button_down[SDLK_p])
		{
			n_iso -= dt;
		}
		if (button_down[SDLK_k])
		{
			side_size += 1;
			update_indices();
			update_vertices();
			send_indices(vbo, ebo, vao, vertices, indices);
		}
		if (button_down[SDLK_l])
		{
			side_size -= 1;
			update_indices();
			update_vertices();
			send_indices(vbo, ebo, vao, vertices, indices);
		}

		update_vertices();
		update_metaballs(dt);

		glClear(GL_COLOR_BUFFER_BIT);
		glClear(GL_DEPTH_BUFFER_BIT);
		SDL_GetWindowSize(window, &width, &height);

		float view[16] =
			{
				float(height) / width,
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
		};

		float transform[16] =
			{
				scale * std::cos(angle1),0.f,scale * (-1) * std::sin(angle1), 0.f,
				0.f,scale,0.f,-0.1f,
				scale * std::sin(angle1) * 0.3f, 0.f,scale * std::cos(angle1) * 0.3f,-0.5f,
				0.f,0.f,0.f,1.f,
		};

		float transform2[16] =
			{
				1.f,
				0.f,
				0.f,
				0.f,
				0.f,
				std::cos(angle2),
				(-1) * std::sin(angle2),
				0.f,
				0.f,
				std::sin(angle2),
				std::cos(angle2),
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
		};

		float transform3[16] =
			{
				std::cos(angle3),
				(-1) * std::sin(angle3),
				0.f,
				0.f,
				std::sin(angle3),
				std::cos(angle3),
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
				0.f,
				0.f,
				0.f,
				0.f,
				1.f,
		};

		// send_indices(vbo, ebo, vao);
		send_vertices(vbo, ebo, vao, vertices);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

		glBindBuffer(GL_ARRAY_BUFFER, ax_vbo);
		glBindVertexArray(ax_vao);

		glDisable(GL_DEPTH_TEST);

		glDrawArrays(GL_LINES, 0, axes.size());

		glEnable(GL_DEPTH_TEST);

		send_iso(iso_vbo, iso_ebo, iso_vao);
		glBindBuffer(GL_ARRAY_BUFFER, iso_vbo);
		glBindVertexArray(iso_vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iso_ebo);
		glDrawElements(GL_LINES, isoline_indices.size(), GL_UNSIGNED_INT, 0);

		glUseProgram(program);
		glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
		glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform);
		glUniformMatrix4fv(transform2_location, 1, GL_TRUE, transform2);
		glUniformMatrix4fv(transform3_location, 1, GL_TRUE, transform3);

		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const &e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
