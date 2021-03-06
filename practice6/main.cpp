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

#define GLM_FORCE_SWIZZLE
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "textures.hpp"

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

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 normal;
out vec2 texcoord;
out vec3 out_position;

void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
	normal = (model * vec4(in_normal, 0.0)).xyz;
	texcoord = in_texcoord;
	out_position = (model * vec4(in_position, 1.0)).xyz;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;

uniform sampler2D albedo_texture;
uniform sampler2D normal_map;
uniform sampler2D ao_map;
uniform sampler2D roughness_map;
uniform vec3 ambient;
uniform vec3 light_position[3];
uniform vec3 light_color[3];
uniform vec3 light_attenuation[3];

in vec2 texcoord;

in vec3 out_position;

out vec4 out_color;

void main()
{
	vec3 result_color = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < 3; ++i)
	{
		vec3 light_vector = light_position[i] - out_position;
		vec3 light_direction = normalize(light_vector);
		vec3 normal = (model * vec4(2*texture(normal_map, texcoord).xyz - 1, 0.0)).xyz;
		float cosine = dot(normal, light_direction);
		float light_factor = max(0.0, cosine);
		float light_distance = length(light_vector);
		float light_intensity = 1.0 / dot(light_attenuation[i],
		vec3(1.0, light_distance, light_distance * light_distance));

		vec3 reflected_directon = normalize(2.0 * normal * dot(normal, light_direction) - light_direction);
		vec3 camera_dir = normalize((inverse(view) * vec4(0, 0, 0, 1)).xyz);
		float specular = pow(max(0.0, dot(reflected_directon, camera_dir)), 4.0) * (1.0 - texture(roughness_map, texcoord).x);

		result_color += (light_factor + specular)* light_intensity * light_color[i];
	}
	out_color = (pow(texture(ao_map, texcoord).x, 4.0) * vec4(ambient, 1.0) + vec4(result_color,1.0)) * texture(albedo_texture, texcoord);
	out_color = vec4(out_color.xyz / (vec3(1.0) + out_color.xyz), 1.0);
	// out_color = vec4(ambient, 1.0) * texture(albedo_texture, texcoord);
}
)";

GLuint create_shader(GLenum type, const char * source)
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

struct vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texcoord;
};

static vertex plane_vertices[]
{
	{{-10.f, -10.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}},
	{{-10.f,  10.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}},
	{{ 10.f, -10.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}},
	{{ 10.f,  10.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}},
};

static std::uint32_t plane_indices[]
{
	0, 1, 2, 2, 1, 3,
};

glm::vec3 light_positions[3] = {
	{1.0, 1.0, 0.0},
	{0.0, 1.0, 1.0},
	{1.0, 1.0, 1.0}
};
glm::vec3 light_colors[3] = {
	{3, 3, 3},
	{3, 3, 3},
	{3, 3, 3}
};
glm::vec3 light_attenuation[3] = {
	{10, 0, 0.1},
	{10, 0, 0.1},
	{10, 0, 0.1}
};

GLuint light_locations[3][3];

int main() try
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

	SDL_Window * window = SDL_CreateWindow("Graphics course practice 5",
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

	glClearColor(0.8f, 0.8f, 1.f, 0.f);

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
	auto program = create_program(vertex_shader, fragment_shader);

	GLuint model_location = glGetUniformLocation(program, "model");
	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint projection_location = glGetUniformLocation(program, "projection");
	GLuint albedo_location = glGetUniformLocation(program, "albedo_texture");
	GLuint normal_location = glGetUniformLocation(program, "normal_map");
	GLuint ao_location = glGetUniformLocation(program, "ao_map");
	GLuint roughness_location = glGetUniformLocation(program, "roughness_map");
	GLuint ambient_location = glGetUniformLocation(program, "ambient");

	light_locations[0][0] = glGetUniformLocation(program, "light_position[0]");
	light_locations[0][1] = glGetUniformLocation(program, "light_position[1]");
	light_locations[0][2] = glGetUniformLocation(program, "light_position[2]");

	light_locations[1][0] = glGetUniformLocation(program, "light_color[0]");
	light_locations[1][1] = glGetUniformLocation(program, "light_color[1]");
	light_locations[1][2] = glGetUniformLocation(program, "light_color[2]");

	light_locations[2][0] = glGetUniformLocation(program, "light_attenuation[0]");
	light_locations[2][1] = glGetUniformLocation(program, "light_attenuation[1]");
	light_locations[2][2] = glGetUniformLocation(program, "light_attenuation[2]");

	glUseProgram(program);
	glUniform1i(albedo_location, 0);
	glUniform1i(normal_location, 1);
	glUniform1i(ao_location, 2);
	glUniform1i(roughness_location, 3);

	GLuint plane_vao, plane_vbo, plane_ebo;
	glGenVertexArrays(1, &plane_vao);
	glBindVertexArray(plane_vao);

	glGenBuffers(1, &plane_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, plane_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(plane_vertices), plane_vertices, GL_STATIC_DRAW);

	glGenBuffers(1, &plane_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(plane_indices), plane_indices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(0));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(12));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)(24));

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	GLuint brick_albedo;
	glGenTextures(1, &brick_albedo);
	glBindTexture(GL_TEXTURE_2D, brick_albedo);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, brick_albedo_width, brick_albedo_height, 0, GL_RGB, GL_UNSIGNED_BYTE, brick_albedo_data);
	glGenerateMipmap(GL_TEXTURE_2D);

	GLuint brick_normal;
	glGenTextures(1, &brick_normal);
	glBindTexture(GL_TEXTURE_2D, brick_normal);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, brick_normal_width, brick_normal_height, 0, GL_RGB, GL_UNSIGNED_BYTE, brick_normal_data);
	glGenerateMipmap(GL_TEXTURE_2D);

	GLuint brick_ao;
	glGenTextures(1, &brick_ao);
	glBindTexture(GL_TEXTURE_2D, brick_ao);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, brick_ao_width, brick_ao_height, 0, GL_RGB, GL_UNSIGNED_BYTE, brick_ao_data);
	glGenerateMipmap(GL_TEXTURE_2D);

	GLuint brick_roughness;
	glGenTextures(1, &brick_roughness);
	glBindTexture(GL_TEXTURE_2D, brick_roughness);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, brick_roughness_width, brick_roughness_height, 0, GL_RGB, GL_UNSIGNED_BYTE, brick_roughness_data);
	glGenerateMipmap(GL_TEXTURE_2D);

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;

	std::map<SDL_Keycode, bool> button_down;

	float view_angle = glm::pi<float>() / 6.f;
	float camera_distance = 15.f;

	bool running = true;
	while (running)
	{
		for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
		{
		case SDL_QUIT:
			running = false;
			break;
		case SDL_WINDOWEVENT: switch (event.window.event)
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

		if (button_down[SDLK_UP])
			camera_distance -= 5.f * dt;
		if (button_down[SDLK_DOWN])
			camera_distance += 5.f * dt;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		float near = 0.1f;
		float far = 100.f;

		glm::mat4 view(1.f);
		view = glm::translate(view, {0.f, 0.f, -camera_distance});
		view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});

		glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

		glUseProgram(program);
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));

		glUniform3f(ambient_location, 0.1, 0.1, 0.1);
		for (int i = 0; i < 3; ++i)
		{
			glUniform3f(light_locations[0][i], 2*(i + 1)*sin(time * (1 - 2*(i % 2))), 4 , 2*(i + 1)*cos(time*(1 - 2*(i % 2))));
			glUniform3f(light_locations[1][i], 1, 1, 1);
			glUniform3f(light_locations[2][i], 0.5, 0.0, 0.1);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, brick_albedo);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, brick_normal);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, brick_ao);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, brick_roughness);

		glm::mat4 model(1.f);
		model = glm::rotate(model, -glm::pi<float>() / 2.f, {1.f, 0.f, 0.f});
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.f);
		model = glm::rotate(model, glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
		model = glm::translate(model, {0.f, 10.f, -10.f});

		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.f);
		model = glm::translate(model, {0.f, 10.f, -10.f});
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		model = glm::mat4(1.f);
		model = glm::translate(model, {10.f, 10.f, 0.f});
		model = glm::rotate(model, -glm::pi<float>() / 2.f, {0.f, 1.f, 0.f});
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));

		glDrawElements(GL_TRIANGLES, std::size(plane_indices), GL_UNSIGNED_INT, nullptr);

		SDL_GL_SwapWindow(window);
	}

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
