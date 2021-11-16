#include "data.hpp"

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

out vec3 position;
out vec3 normal;
out vec2 texcoord;

void main()
{
	gl_Position = projection * view * model * vec4(in_position, 1.0);
	position = (model * vec4(in_position, 1.0)).xyz;
	normal = normalize((model * vec4(in_normal, 0.0)).xyz);
	texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 ambient;

uniform vec3 light_direction;
uniform vec3 sun_light_color;

uniform mat4 transform;
uniform mat4 model;

uniform sampler2D shadow_map;
uniform sampler2D albedo_texture;
uniform sampler2D normal_map;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

layout (location = 0) out vec4 out_color;

uniform vec3 light_position[3];
uniform vec3 light_color[3];
uniform vec3 light_attenuation[3];

void main()
{
	vec4 shadow_pos = transform * vec4(position, 1.0);
	shadow_pos /= shadow_pos.w;
	shadow_pos = shadow_pos * 0.5 + vec4(0.5);

	// bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
	// float shadow_factor = 1.0;
	// if (in_shadow_texture)
	// 	shadow_factor = texture(shadow_map, shadow_pos.xyz - vec3(0.f, 0.f, 0.008f));

	vec2 data = vec2(0.0);
	float sum_w = 0.0;
	const int N = 7;
	float radius = 5.0;
	for (int x = -N; x <= N; ++x) {
		for (int y = -N; y <= N; ++y) {
			float c = exp(-float(x*x + y*y) / (radius*radius));
			data += c * texture(shadow_map, shadow_pos.xy +
			vec2(x,y) / vec2(textureSize(shadow_map, 0))).rg;
			sum_w += c;
		}
	}

	data /= sum_w;

	// vec2 data = texture(shadow_map, shadow_pos.xy).rg;
	float mu = data.r;
	float sigma = data.g - mu * mu;
	float z = shadow_pos.z;
	float shadow_factor = (z < mu) ? 1.0 : sigma / (sigma + (z-mu)*(z-mu));
	// shadow_factor = (shadow_factor < 0.125) ? 0.0 : 1.0;

	vec3 albedo = vec3(1.0, 1.0, 1.0);

	// float shadow_factor = 1.0;

	vec3 map_normal = texture(normal_map, texcoord).xyz;
	// map_normal = vec3(dot(vec3(1.0, 1.0, 0.0), map_normal), dot(vec3(0.0, 1.0, 1.0), map_normal), dot(vec3(0.0, 0.0, 1.0), map_normal));
	if (length(map_normal) < 0.1)
	{
		map_normal = vec3(0.0, 0.0, 1.0);
	}
	else
	{
		map_normal = 2*map_normal - 1;
	}
	

	map_normal = normalize(map_normal);

	mat3 normal_transform = mat3(cross(vec3(1.0, 1.0, 1.0), normal), cross(vec3(1.0, 2.0, 3.0), normal), normal);

	map_normal = normalize(normal_transform * map_normal);

	// map_normal = normal;

	// map_normal = (model * vec4(map_normal, 1.0f)).xyz;

	// vec3 map_normal = normal;
	vec3 result_color = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < 3; ++i)
	{
		vec3 light_vector = (model * vec4(light_position[i], 1.0f)).xyz - position;
		light_vector = light_vector;
		vec3 light_direction = normalize(light_vector);
		float cosine = dot(map_normal, light_direction);
		float light_factor = max(0.0, cosine);
		float light_distance = length(light_vector);
		float light_intensity = 1.0 / dot(light_attenuation[i],
		vec3(1.0, light_distance, light_distance * light_distance));

		// vec3 reflected_directon = normalize(2.0 * map_normal * dot(map_normal, light_direction) - light_direction);
		// vec3 camera_dir = normalize((inverse(view) * vec4(0, 0, 0, 1)).xyz);
		// float specular = pow(max(0.0, dot(reflected_directon, camera_dir)), 4.0) * (1.0 - texture(roughness_map, texcoord).x);

		result_color += light_factor* light_intensity * light_color[i];
		// result_color += vec3(light_intensity);
	}

	vec3 light = ambient;
	light += (sun_light_color * max(0.0, dot(map_normal, light_direction))) * shadow_factor;
	// light += result_color;
	vec3 color = albedo * light;

	out_color = vec4(color + result_color, 1.0) * texture(albedo_texture, texcoord);
	// out_color = texture(albedo_texture, texcoord);
	// out_color = vec4(result_color, 1.0);
	// out_color = vec4(out_color.xyz / (vec3(1.0) + out_color.xyz), 1.0);
	// out_color = vec4((map_normal + 1.0) / 2.0, 1.0);
}
)";

const char debug_vertex_shader_source[] =
R"(#version 330 core

vec2 vertices[6] = vec2[6](
	vec2(-1.0, -1.0),
	vec2( 1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0, -1.0),
	vec2( 1.0,  1.0),
	vec2(-1.0,  1.0)
);

out vec2 texcoord;

void main()
{
	vec2 position = vertices[gl_VertexID];
	gl_Position = vec4(position * 0.25 + vec2(-0.75, -0.75), 0.0, 1.0);
	texcoord = position * 0.5 + vec2(0.5);
}
)";

const char debug_fragment_shader_source[] =
R"(#version 330 core

uniform sampler2D shadow_map;

in vec2 texcoord;

layout (location = 0) out vec4 out_color;

void main()
{
	// vec2 coord = texcoord;

	// if (texcoord.x > 0.5)
	// {
	// 	if (texcoord.y > 0.5)
	// 	{
	// 		out_color = texture(shadow_map, vec3(0.0f) - vec3(coord * 4 - 3, 1.0f));
	// 	}
	// 	else
	// 	{
	// 		out_color = texture(shadow_map, vec3(0.0f) - vec3(coord.x * 4 - 3, coord.y*4 - 1, -1.0f));
	// 	}
	// }
	// else
	// {
	// 	if (texcoord.y > 0.5)
	// 	{
	// 		out_color = texture(shadow_map, vec3(0.0f) - vec3(1.0f, coord.x*4 - 1, coord.y*4 - 3));
	// 	}
	// 	else
	// 	{
	// 		out_color = texture(shadow_map, vec3(0.0f) - vec3(-1.0f, coord.x*4 - 1, coord.y*4 - 1));
	// 	}
	// }

	out_color = texture(shadow_map, texcoord);
}
)";

const char shadow_vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;

void main()
{
	gl_Position = transform * model * vec4(in_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
R"(#version 330 core
out vec4 depth;
void main()
{
	float z = gl_FragCoord.z + 0.008;
	float dzx = dFdx(z);
	float dzy = dFdy(z);
	depth = vec4(z, z * z + 0.25 * (dzx*dzx + dzy*dzy), 0.0, 0.0);
}
)";

const char mirror_vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 center;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 position;
out vec3 normal;
out vec2 texcoord;

void main()
{
	gl_Position = projection * view * model * vec4(in_position + center, 1.0);
	position = (model * vec4(in_position + center, 1.0)).xyz;
	normal = normalize((model * vec4(in_normal, 0.0)).xyz);
	texcoord = in_texcoord;
}
)";

const char mirror_fragment_shader_source[] =
R"(#version 330 core

uniform mat4 model;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

uniform samplerCube mirror_texture;
uniform vec3 center;
uniform vec3 camera_position;

layout (location = 0) out vec4 out_color;

void main()
{
	vec3 dir = normalize(position - (model*vec4(center, 1.f)).xyz);
	vec3 camera_dir = normalize(camera_position - position);
	vec3 reflected = 2.0 * normal * dot(normal, camera_dir) - camera_dir;

	out_color = texture(mirror_texture, reflected);
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

std::pair<glm::vec3, glm::vec3> bbox(std::vector<vertex> const & vertices)
{
	static const float inf = std::numeric_limits<float>::infinity();

	glm::vec3 min = glm::vec3( inf);
	glm::vec3 max = glm::vec3(-inf);

	for (auto const & v : vertices)
	{
		min = glm::min(min, v.position);
		max = glm::max(max, v.position);
	}

	return {min, max};
}


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
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_Window * window = SDL_CreateWindow("Graphics course practice 9",
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

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);


	auto program = create_program(vertex_shader, fragment_shader);

	GLuint model_location = glGetUniformLocation(program, "model");
	GLuint view_location = glGetUniformLocation(program, "view");
	GLuint projection_location = glGetUniformLocation(program, "projection");
	GLuint transform_location = glGetUniformLocation(program, "transform");

	GLuint ambient_location = glGetUniformLocation(program, "ambient");
	GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
	GLuint sun_light_color_location = glGetUniformLocation(program, "sun_light_color");

	GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
	GLuint texture_location = glGetUniformLocation(program, "albedo_texture");
	GLuint normal_location = glGetUniformLocation(program, "normal_map");

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
	glUniform1i(texture_location, 0);
	glUniform1i(shadow_map_location, 1);
	glUniform1i(normal_location, 2);

	auto debug_vertex_shader = create_shader(GL_VERTEX_SHADER, debug_vertex_shader_source);
	auto debug_fragment_shader = create_shader(GL_FRAGMENT_SHADER, debug_fragment_shader_source);
	auto debug_program = create_program(debug_vertex_shader, debug_fragment_shader);

	GLuint debug_shadow_map_location = glGetUniformLocation(debug_program, "shadow_map");

	glUseProgram(debug_program);
	glUniform1i(debug_shadow_map_location, 0);

	auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
	auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
	auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

	GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
	GLuint shadow_transform_location = glGetUniformLocation(shadow_program, "transform");

	auto mirror_vertex_shader = create_shader(GL_VERTEX_SHADER, mirror_vertex_shader_source);
	auto mirror_fragment_shader = create_shader(GL_FRAGMENT_SHADER, mirror_fragment_shader_source);
	auto mirror_program = create_program(mirror_vertex_shader, mirror_fragment_shader);

	GLuint mirror_texture_location = glGetUniformLocation(mirror_program, "mirror_texture");
	GLuint mirror_model_location = glGetUniformLocation(mirror_program, "model");
	GLuint mirror_view_location = glGetUniformLocation(mirror_program, "view");
	GLuint mirror_projection_location = glGetUniformLocation(mirror_program, "projection");
	GLuint mirror_center_location = glGetUniformLocation(mirror_program, "center");
	GLuint camera_position_location = glGetUniformLocation(mirror_program, "camera_position");

	glUseProgram(mirror_program);
	glUniform1i(mirror_texture_location, 0);


	// std::vector<RenderObject> render_objects;
	// {
	// 	// std::ifstream sponza_file(PRACTICE_SOURCE_DIRECTORY "/sponza/sponza.obj");
	// 	// std::ifstream mtl_file(PRACTICE_SOURCE_DIRECTORY "/sponza/sponza.mtl");
	// 	render_objects = load_objects(PRACTICE_SOURCE_DIRECTORY "/sponza/sponza.obj", PRACTICE_SOURCE_DIRECTORY "/sponza/");
	// }
	// fill_normals(vertices, indices);

	RenderScene scene(PRACTICE_SOURCE_DIRECTORY "/sponza/sponza.obj", PRACTICE_SOURCE_DIRECTORY "/sponza/");

	GLuint debug_vao;
	glGenVertexArrays(1, &debug_vao);

	GLsizei shadow_map_resolution = 2048;

	GLuint shadow_map;
	glGenTextures(1, &shadow_map);
	glBindTexture(GL_TEXTURE_2D, shadow_map);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	// glTexParameterf(GL_TEXTURE_2D, )
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, shadow_map_resolution, shadow_map_resolution, 0, GL_RGBA, GL_FLOAT, nullptr);

	GLuint depth_map;
	glGenTextures(1, &depth_map);
	glBindTexture(GL_TEXTURE_2D, depth_map);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

	GLuint shadow_fbo;
	glGenFramebuffers(1, &shadow_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_map, 0);

	Mirror mirror(512);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		throw std::runtime_error("Incomplete framebuffer!");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	auto last_frame_start = std::chrono::high_resolution_clock::now();

	float time = 0.f;
	bool paused = false;

	std::map<SDL_Keycode, bool> button_down;

	float view_elevation = glm::radians(45.f);
	float view_azimuth = 0.f;
	float camera_distance = 0.5f;
	float camera_target = 0.05f;
	float x_delta = 0;
	float z_delta = 0;
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

			if (event.key.keysym.sym == SDLK_SPACE)
				paused = !paused;

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
		if (!paused)
			time += dt;

		if (button_down[SDLK_UP])
			camera_distance -= 1.f * dt;
		if (button_down[SDLK_DOWN])
			camera_distance += 1.f * dt;

		if (button_down[SDLK_LEFT])
			view_azimuth -= 2.f * dt;
		if (button_down[SDLK_RIGHT])
			view_azimuth += 2.f * dt;

		if (button_down[SDLK_w])
			x_delta += 1.f * dt;
		
		if (button_down[SDLK_s])
			x_delta -= 1.f * dt;

		if (button_down[SDLK_a])
			z_delta += 1.f * dt;

		if (button_down[SDLK_d])
			z_delta -= 1.f * dt;

		if (button_down[SDLK_u])
			mirror.center[0] += 0.5f * dt;

		if (button_down[SDLK_j])
			mirror.center[0] -= 0.5f * dt;

		if (button_down[SDLK_h])
			mirror.center[2] += 0.5f * dt;

		if (button_down[SDLK_k])
			mirror.center[2] -= 0.5f * dt;

		if (button_down[SDLK_y])
			mirror.center[1] += 0.5f * dt;

		if (button_down[SDLK_i])
			mirror.center[1] -= 0.5f * dt;

		std::cout << camera_distance << " " << view_azimuth << std::endl;

		glm::mat4 model(1.f);
		// model = glm::translate(model, {x_delta, 0.0, z_delta});

		// glm::vec3 light_direction = glm::normalize(glm::vec3(std::cos(time * 0.5f), 1.f, std::sin(time * 0.5f)));
		glm::vec3 light_direction = glm::normalize(glm::vec3(0.f, 1.f, 0.f));

		glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, shadow_map_resolution, shadow_map_resolution);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glm::vec3 light_z = -light_direction;
		glm::vec3 light_x = glm::normalize(glm::cross(light_z, {1.f, 0.f, 0.f}));
		glm::vec3 light_y = glm::cross(light_x, light_z);

		auto [ min_v, max_v ] = bbox(scene.vertices);
		std::vector<glm::vec3> min_max_v = {min_v, max_v};

		glm::vec3 c = (min_v + max_v) * 0.5f; 

		float shadow_scale_x = -std::numeric_limits<float>::infinity();
		float shadow_scale_y = -std::numeric_limits<float>::infinity();
		float shadow_scale_z = -std::numeric_limits<float>::infinity();

		for (int i = 0; i < 8; ++i)
		{
			float x, y, z;

			x = min_max_v[i % 2 > 0][0];
			y = min_max_v[i % 4 > 1][1];
			z = min_max_v[i % 8 > 3][2];

			glm::vec3 v = {x,y,z};

			shadow_scale_x = std::max(shadow_scale_x, abs(glm::dot(v - c, light_x)));
			shadow_scale_y = std::max(shadow_scale_y, abs(glm::dot(v - c, light_y)));
			shadow_scale_z = std::max(shadow_scale_z, abs(glm::dot(v - c, light_z)));
		}

		glm::mat4 transform = glm::mat4(1.f);
		for (size_t i = 0; i < 3; ++i)
		{
			transform[i][0] = light_x[i] / shadow_scale_x;
			transform[i][1] = light_y[i] / shadow_scale_y;
			transform[i][2] = light_z[i] / shadow_scale_z;
		}

		transform[3][0] = -glm::dot(c, light_x / shadow_scale_x);
		transform[3][1] = -glm::dot(c, light_y / shadow_scale_y);
		transform[3][2] = -glm::dot(c, light_z / shadow_scale_z); 

		glUseProgram(shadow_program);
		glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
		glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadow_map);
		
		scene.render();

		glGenerateMipmap(GL_TEXTURE_2D);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, width, height);

		glClearColor(0.8f, 0.8f, 0.9f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		float near = 0.01f;
		float far = 10.f;

		glm::mat4 view(1.f);
		view = glm::translate(view, {0.f, 0.f, -camera_distance});
		view = glm::rotate(view, view_elevation, {1.f, 0.f, 0.f});
		view = glm::rotate(view, view_azimuth, {0.f, 1.f, 0.f});
		view = glm::translate(view, {x_delta, -camera_target, z_delta});

		glm::mat4 projection = glm::mat4(1.f);
		projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

		glUseProgram(program);
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
		glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

		glUniform3f(ambient_location, 0.1f, 0.1f, 0.1f);
		glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
		glUniform3f(sun_light_color_location, 0.8f, 0.8f, 0.8f);

		for (int i = 0; i < 3; ++i)
		{
			glUniform3f(light_locations[0][i], 0.5 * (i - 1), 0.1 , 0.2 * (i - 1));
			// glUniform3f(light_locations[1][i], (i == 0), (i == 1), (i == 2));
			glUniform3f(light_locations[1][i], 1.0, 1.0, 1.0);
			glUniform3f(light_locations[2][i], 1, 0, 100);
		}

		
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, shadow_map);

		mirror.cubemap.render(projection_location, scene, view_location, model_location, mirror.center);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, width, height);


		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
		glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
		glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
		scene.render();

		glUseProgram(debug_program);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadow_map);
		glBindVertexArray(debug_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

		glUseProgram(mirror_program);
		glUniformMatrix4fv(mirror_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
		glUniformMatrix4fv(mirror_view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
		glUniformMatrix4fv(mirror_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
		glUniform3fv(mirror_center_location, 1, reinterpret_cast<float *>(&mirror.center));
		glUniform3fv(camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
		mirror.render();

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
