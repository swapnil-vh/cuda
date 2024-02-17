//Header file inclusion area
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<memory.h>
#include<math.h>

#include<X11/Xlib.h>
#include<X11/Xutil.h>
#include<X11/XKBlib.h>
#include<X11/keysym.h>

#include<GL/glew.h>
#include<GL/gl.h>
#include<GL/glx.h>
#include "vmath.h"
#include <chrono>

//namespaces
using namespace std;
using namespace vmath;

//global enums and structure and Union declaration area
enum ATTRIBUTE_INDEX{
	SPV_ATTRIBUTE_POSITION = 0,
	SPV_ATTRIBUTE_COLOR,
	SPV_ATTRIBUTE_NORMAL,
	SPV_ATTRIBUTE_TEXCOORD
};

//global variables declaration area
bool SPV_gbFullScreen = false;
Display *SPV_gpDisplay  = NULL;
XVisualInfo *SPV_gpXVisualInfo = NULL;
GLXContext SPV_gGLXContext;

typedef GLXContext (*glxCreateContextAttribsArbProc)(Display *, GLXFBConfig, GLXContext, Bool, const int *);
glxCreateContextAttribsArbProc glxCreateContextAttribsArb = NULL;
GLXFBConfig SPV_gGLXFBConfig;

Colormap SPV_gColorMap;
Window SPV_gWindow;
int SPV_giWindowWidth = 900;
int SPV_giWindowHeight = 600;
static const char *SPV_appName = "Swapnil Vhanale: Sine Wave Grid";
static char appName[256];
static unsigned int gfps = 0;

FILE *SPV_debug;

GLuint SPV_gVertexShaderObject;
GLuint SPV_gFragmentShaderObject;
GLuint SPV_gShaderProgramObject;

mat4 SPV_projectionMatrix;
GLfloat color[][3] = {{1.0f, 0.0f, 0.0f},
					  {0.0f, 1.0f, 0.0f},
					  {0.0f, 0.0f, 1.0f},
					  {1.0f, 0.5f, 0.0f},
					};
static int colorIndex = 0;

GLuint SPV_uProjectionMatrixUniform;
GLuint uColorUniform;
GLuint SPV_vao_pyramid;
GLuint SPV_vbo_pyramid_position;
GLuint SPV_vbo_pyramid_color;

unsigned int meshWidth = 128;
unsigned int meshHeight = 128;

#define MY_ARRAY (meshWidth * meshHeight * 4)

float *pos = nullptr;
float *devVertextArray = nullptr;
float animTime = 0.0f;

bool bGPU = false;

//entry point function
int main(void)
{
	//function prototype
	void CreateWindow(void);
	void toggleFullScreen(void);
	void initialize(void);
	void resize(int, int);
	void display(void);
	void uninitialize(void);

	//local variables declarations
	int SPV_winWidth = SPV_giWindowWidth;
	int SPV_winHeight = SPV_giWindowHeight;
	bool SPV_bDone = false;

	memset(appName, 0, 256);
	//code
	SPV_debug = fopen("./debug.txt", "w");
	if(SPV_debug == NULL)
	{
		printf("Error: Debug file opening error \nExiting now \n");
		exit(1);
	}

	CreateWindow();
	initialize();

	//message loop
	XEvent SPV_event;
	KeySym SPV_keysym;

	while(SPV_bDone == false)
	{
		while(XPending(SPV_gpDisplay))
		{
			XNextEvent(SPV_gpDisplay, &SPV_event);
			switch(SPV_event.type)
			{
				case MapNotify:
					break;

				case KeyPress:
					SPV_keysym = XkbKeycodeToKeysym(SPV_gpDisplay, SPV_event.xkey.keycode, 0, 0);
					switch(SPV_keysym)
					{
						case XK_Escape:
							SPV_bDone = true;
							break;

						case XK_f:
						case XK_F:
							if(SPV_gbFullScreen == false)
							{
								toggleFullScreen();
								SPV_gbFullScreen = true;
							}
							else
							{
								toggleFullScreen();
								SPV_gbFullScreen = false;
							}
							break;
						
						case XK_i :
							meshWidth += 128;
							meshHeight += 128;
							if(meshWidth > 2048)
								meshWidth = 2048;
							if(meshHeight > 2048)
								meshHeight = 2048;

							printf("Rendering %u x %u size grid\n", meshWidth, meshHeight);							
							break;

						case XK_d :
							meshWidth -= 128;
							meshHeight -= 128;
							if(meshWidth < 128)
								meshWidth = 128;
							if(meshHeight < 128)
								meshHeight = 128;
							
							printf("Rendering %u x %u size grid\n", meshWidth, meshHeight);
							break;

						case XK_c :
							printf("Rendering with CPU computing mode\n");
							bGPU = false;
							break;
						
						case XK_g :
							printf("Rendering With GPU computing mode\n");
							bGPU = true;
							break;

						case XK_k :
							colorIndex++;
							if(colorIndex > 3) colorIndex = 0;
							break;

						default:
							break;
					}
					break;

				case ButtonPress:
					switch(SPV_event.xbutton.button)
					{
						case 1:
							break;
						case 2:
							break;
						case 3:
							break;
						default:
							break;
					}
					break;

				case MotionNotify:
					break;

				case ConfigureNotify:
					SPV_winWidth = SPV_event.xconfigure.width; 
					SPV_winHeight = SPV_event.xconfigure.height;
					resize(SPV_winWidth, SPV_winHeight);
					break;

				case Expose:
					break; 

				case DestroyNotify:
					break;

				case 33:
					SPV_bDone = true;
					break;

				default:
					break;
			}
		}

		sprintf(appName, "%s  |  Grid dimention:[%u x %u]  |  Computing Mode:%s  |  FPS:%u\0",SPV_appName, meshWidth, meshHeight, bGPU ? "GPU" : "CPU", gfps);
		XStoreName(SPV_gpDisplay, SPV_gWindow, appName);
		display();
	}

	uninitialize();
	return(0);
}

void CreateWindow(void)
{
	//function prototype declaration area
	void uninitialize(void);

	//local variables declaration area
	XSetWindowAttributes SPV_winAttribs;
	GLXFBConfig *SPV_pGLXFBConfig = NULL;
	GLXFBConfig SPV_bestGLXFBConfig;
	XVisualInfo *SPV_pTempXVisualInfo = NULL;
	int SPV_numFBConfigs;
	int SPV_defaultScreen;
	int SPV_styleMask;
	static int SPV_frameBufferAttributes[] = {
		GLX_X_RENDERABLE, true,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_STENCIL_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_DOUBLEBUFFER, True,
		None
	};

	SPV_gpDisplay = XOpenDisplay(NULL);
	if(SPV_gpDisplay == NULL)
	{
		printf("Error: Unable to Open X Display \n Exiting now..\n");
		uninitialize();
		exit(1);
	}

	SPV_defaultScreen = XDefaultScreen(SPV_gpDisplay);

	SPV_pGLXFBConfig = glXChooseFBConfig(SPV_gpDisplay, SPV_defaultScreen, SPV_frameBufferAttributes, &SPV_numFBConfigs);
	printf("Found number of FBConfigs : %d \n", SPV_numFBConfigs);

	int SPV_bestFrameBufferConfig = -1;
	int SPV_worstFrameBufferConfig = -1;
	int SPV_bestNumberOfSamples = -1;
	int SPV_worstNumberOfSamples = 999;

	for(int i=0; i<SPV_numFBConfigs; i++)
	{
		SPV_pTempXVisualInfo = glXGetVisualFromFBConfig(SPV_gpDisplay, SPV_pGLXFBConfig[i]);
		if(SPV_pTempXVisualInfo != NULL)
		{
			int sampleBuffers, samples;
			glXGetFBConfigAttrib(SPV_gpDisplay, SPV_pGLXFBConfig[i], GLX_SAMPLE_BUFFERS, &sampleBuffers);
			glXGetFBConfigAttrib(SPV_gpDisplay, SPV_pGLXFBConfig[i], GLX_SAMPLES, &samples);
			if((SPV_bestFrameBufferConfig < 0) || sampleBuffers && (samples > SPV_bestNumberOfSamples))
			{
				SPV_bestFrameBufferConfig = i;
				SPV_bestNumberOfSamples = samples;
			}
			
			if((SPV_worstFrameBufferConfig < 0) || (!sampleBuffers) || (samples < SPV_worstNumberOfSamples))
			{
				SPV_worstFrameBufferConfig = i;
				SPV_worstNumberOfSamples = samples;
			}

			printf("For FBConfig[%d] : XVIsualID = %lu, SampleBuffers = %d, Samples = %d \n", i, SPV_pTempXVisualInfo->visualid, sampleBuffers, samples);
		}

		XFree(SPV_pTempXVisualInfo);
	}

	printf("The best GLXFBConfig is : %d\n", SPV_bestFrameBufferConfig);

	SPV_bestGLXFBConfig = SPV_pGLXFBConfig[SPV_bestFrameBufferConfig];
	SPV_gGLXFBConfig = SPV_bestGLXFBConfig;
	XFree(SPV_pGLXFBConfig);

	SPV_gpXVisualInfo = glXGetVisualFromFBConfig(SPV_gpDisplay, SPV_gGLXFBConfig);

	SPV_winAttribs.border_pixel = 0;
	SPV_winAttribs.background_pixmap = 0;
	SPV_winAttribs.colormap = XCreateColormap(SPV_gpDisplay, 
			RootWindow(SPV_gpDisplay, SPV_gpXVisualInfo->screen),
			SPV_gpXVisualInfo->visual,
			AllocNone);

	SPV_gColorMap = SPV_winAttribs.colormap;
	SPV_winAttribs.background_pixel = BlackPixel(SPV_gpDisplay, SPV_defaultScreen);
	SPV_winAttribs.event_mask = ExposureMask | VisibilityChangeMask	 | ButtonPressMask | KeyPressMask | PointerMotionMask | StructureNotifyMask;
	SPV_styleMask = CWBorderPixel | CWBackPixel | CWEventMask | CWColormap;

	SPV_gWindow = XCreateWindow(SPV_gpDisplay,
			RootWindow(SPV_gpDisplay, SPV_gpXVisualInfo->screen),
			0,
			0,
			SPV_giWindowWidth,
			SPV_giWindowHeight,
			0,
			SPV_gpXVisualInfo->depth,
			InputOutput,
			SPV_gpXVisualInfo->visual,
			SPV_styleMask,
			&SPV_winAttribs);

	if(!SPV_gWindow)
	{
		printf("Error: Failed to create main window \n Exiting now\n");
		uninitialize();
		exit(1);
	}

	sprintf(appName, "%s  |  Grid dimention:[%u x %u]  |  Computing Mode:%s  |  FPS:%u\0",SPV_appName, meshWidth, meshHeight, bGPU ? "GPU" : "CPU", gfps);
	XStoreName(SPV_gpDisplay, SPV_gWindow, appName);

	Atom SPV_windowManagerDelete = XInternAtom(SPV_gpDisplay, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(SPV_gpDisplay, SPV_gWindow, &SPV_windowManagerDelete, 1);
	XMapWindow(SPV_gpDisplay, SPV_gWindow);
}

void toggleFullScreen(void)
{
	//local variables declaration
	Atom SPV_wmState;
	Atom SPV_fullScreen;
	XEvent SPV_xev = {0};

	//code
	SPV_wmState = XInternAtom(SPV_gpDisplay, "_NET_WM_STATE", False);
	memset(&SPV_xev, 0, sizeof(SPV_xev));

	SPV_xev.type = ClientMessage;
	SPV_xev.xclient.window = SPV_gWindow;
	SPV_xev.xclient.message_type = SPV_wmState;
	SPV_xev.xclient.format = 32;
	SPV_xev.xclient.data.l[0] = SPV_gbFullScreen ? 0 : 1;
	SPV_fullScreen = XInternAtom(SPV_gpDisplay, "_NET_WM_STATE_FULLSCREEN", False);
	SPV_xev.xclient.data.l[1] = SPV_fullScreen;

	XSendEvent(SPV_gpDisplay,
			  RootWindow(SPV_gpDisplay, SPV_gpXVisualInfo->screen),
			  False,
			  StructureNotifyMask,
			  &SPV_xev);
}

void initialize(void)
{
	//function prototype
	void resize(int, int);
	void uninitialize(void);
	int checkShaderCompilationStatus(GLuint, const char *);
	int checkShaderProgramLinkStatus(GLuint, const char *);

	//local variables declarations
	const int SPV_attribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 5,
		GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
		None};

	bool SPV_isDirectContext = false;

	//code
	glxCreateContextAttribsArb = (glxCreateContextAttribsArbProc)glXGetProcAddressARB((GLubyte *)"glXCreateContextAttribsARB");
	SPV_gGLXContext = glxCreateContextAttribsArb(SPV_gpDisplay, SPV_gGLXFBConfig, 0, True, SPV_attribs);
	
	if(!SPV_gGLXContext)
	{
		const int lowestAttribs[] = { GLX_CONTEXT_MAJOR_VERSION_ARB, 1,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None};

		SPV_gGLXContext = glxCreateContextAttribsArb(SPV_gpDisplay, SPV_gGLXFBConfig, 0, True, lowestAttribs);
		printf("Received lowest rendering context available from driver \n");
	}
	else
	{
		printf("Received the specified context from driver ..\n");
	}

	SPV_isDirectContext = glXIsDirect(SPV_gpDisplay, SPV_gGLXContext);
	if(SPV_isDirectContext)
	{
		printf("The rendering context is direct H/W rendering supported context ..\n");
	}
	else
	{
		printf("The rendering context is S / W rendering supporting context ..\n");
	}
	
	glXMakeCurrent(SPV_gpDisplay, SPV_gWindow, SPV_gGLXContext);


	GLenum glew_error = glewInit();
	if(glew_error != GLEW_OK)
	{
		printf("glew init failed .. Exiting now .. \n");
		uninitialize();
		exit(1);
	}

	fprintf(SPV_debug, "OpenGL Vendor: %s \n", glGetString(GL_VENDOR));
	fprintf(SPV_debug, "OpenGL renderer Name : %s \n", glGetString(GL_RENDERER));
	fprintf(SPV_debug, "OpenGL Version: %s \n", glGetString(GL_VERSION));
	fprintf(SPV_debug, "GLSL version: %s \n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	GLint SPV_numExtensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &SPV_numExtensions);
	for(int i =0; i < SPV_numExtensions; i++)
	{
		fprintf(SPV_debug, "%s \n", glGetStringi(GL_EXTENSIONS, i));
	}

	//Vertex Shader
	SPV_gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const char *SPV_vertexShaderSourceCode = "#version 450\n" \
											  "in vec4 vPosition;\n" \
											  "uniform vec3 vColor;\n" \
											  "uniform mat4 u_mvpMatrix;\n" \
											  "out vec3 out_color;\n" \
											  "void main(void)\n" \
											  "{\n" \
											  "		gl_Position = u_mvpMatrix * vPosition;\n" \
											  "		out_color = vColor;\n" \
											  "}";
	glShaderSource(SPV_gVertexShaderObject, 1, (const char **)&SPV_vertexShaderSourceCode, NULL);
	glCompileShader(SPV_gVertexShaderObject);
	if(checkShaderCompilationStatus(SPV_gVertexShaderObject, "Vertex Shader ") != 0)
	{
		uninitialize();
		exit(1);
	}

	//Fragment shader
	SPV_gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
	const char *SPV_gFragmentShaderSourceCode = "#version 450\n" \
												"in vec3 out_color;\n"	\
												"out vec4 FragColor;\n" \
												"void main(void)\n" \
												"{\n"	\
												"	FragColor = vec4(out_color, 1.0f);\n" \
												"}";
	glShaderSource(SPV_gFragmentShaderObject, 1, (const char **)&SPV_gFragmentShaderSourceCode, NULL);
	glCompileShader(SPV_gFragmentShaderObject);
	if(checkShaderCompilationStatus(SPV_gFragmentShaderObject, "Fragment Shader ") != 0)
	{
		uninitialize();
		exit(1);
	}

	//Shader programObject
	SPV_gShaderProgramObject = glCreateProgram();
	glAttachShader(SPV_gShaderProgramObject, SPV_gVertexShaderObject);
	glAttachShader(SPV_gShaderProgramObject, SPV_gFragmentShaderObject);
	glBindAttribLocation(SPV_gShaderProgramObject, SPV_ATTRIBUTE_POSITION, "vPosition");
	//glBindAttribLocation(SPV_gShaderProgramObject, SPV_ATTRIBUTE_COLOR, "vColor");
	glLinkProgram(SPV_gShaderProgramObject);
	if(checkShaderProgramLinkStatus(SPV_gShaderProgramObject, "Shader Program Link ") != 0)
	{
		uninitialize();
		exit(1);
	}

	SPV_uProjectionMatrixUniform = glGetUniformLocation(SPV_gShaderProgramObject, "u_mvpMatrix");
	uColorUniform = glGetUniformLocation(SPV_gShaderProgramObject, "vColor");

	//VAO creation area
	glGenVertexArrays(1, &SPV_vao_pyramid);
	glBindVertexArray(SPV_vao_pyramid);

	glGenBuffers(1, &SPV_vbo_pyramid_position);
	glBindBuffer(GL_ARRAY_BUFFER, SPV_vbo_pyramid_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MY_ARRAY, NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(SPV_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SPV_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//Depth related lines
	glClearDepth(1.0f);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	SPV_projectionMatrix = mat4::identity();
	
	resize(SPV_giWindowWidth, SPV_giWindowHeight);

	auto cudaOp = cudaMalloc(&devVertextArray, sizeof(float)*MY_ARRAY);
	if(cudaOp != cudaError::cudaSuccess)
	{
		uninitialize();
		exit(1);
	}
}


void resize(int width, int height)
{
	//code
	if(height <=0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	SPV_projectionMatrix = vmath::perspective(45.0f, (GLfloat)width/(GLfloat)height, 0.1f, 100.0f);
}

void launchCPUKernel(unsigned int width, unsigned int height, float time)
{
	for(int i=0; i<width; i++)
	{
		for(int j =0; j < height; j++)
		{	
			//for(int k=0; k<4; k++)
			{
				float u = i / (float)width;
				float v = j / (float)height;
				float frequency = 4.0f;
				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;
				float w = sinf(u * frequency + time) * cosf(v * frequency + time) * 0.2f;
				pos[width * i * 4 + j * 4 + 0] = u;
				pos[width * i * 4 + j * 4 + 1] = w;
				pos[width * i * 4 + j * 4 + 2] = v;
				pos[width * i * 4 + j * 4 + 3] = 1.0;
			}
		}
	}
}

// __global__ void launchGPUKernel(unsigned int width, unsigned int height, float time, float *vertexArray)
// {
// 	unsigned int tid = blockIdx.x;
// 	unsigned int limit = width * height;
// 	if(tid < limit)
// 	{
// 		int x = tid % width;
// 		int y = tid / height;
// 		float u = x / (float)width;
// 		float v = y / (float) height;
// 		float frequency = 4.0f;
// 		u = u * 2.0f - 1.0f;
// 		v = v * 2.0f - 1.0f;
// 		float w = sinf(u * frequency + time) * cosf(v * frequency + time) * 0.2f;

// 		vertexArray[width * y * 4 + x * 4 + 0] = u;
// 		vertexArray[width * y * 4 + x * 4 + 1] = w;
// 		vertexArray[width * y * 4 + x * 4 + 2] = v;
// 		vertexArray[width * y * 4 + x * 4 + 3] = 1.0;
// 	}
// }

__global__ void launchGPUKernel(unsigned int width, unsigned int height, float time, float *vertexArray)
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int limit = width * height;
	if(tid < limit)
	{
		int x = tid % width;
		int y = tid / height;
		float u = x / (float)width;
		float v = y / (float) height;
		float frequency = 4.0f;
		u = u * 2.0f - 1.0f;
		v = v * 2.0f - 1.0f;
		float w = sinf(u * frequency + time) * cosf(v * frequency + time) * 0.2f;

		vertexArray[width * y * 4 + x * 4 + 0] = u;
		vertexArray[width * y * 4 + x * 4 + 1] = w;
		vertexArray[width * y * 4 + x * 4 + 2] = v;
		vertexArray[width * y * 4 + x * 4 + 3] = 1.0;
	}
}
void display(void)
{
	void uninitialize(void);

	//local variables declaration area
	static auto startTime = std::chrono::high_resolution_clock::now();
	static unsigned int fps = 0;
	static unsigned int curGridSize = 0;
	mat4 SPV_modelMatrix = vmath::translate(0.0f, 0.0f, -2.0f);
	mat4 SPV_viewMatrix = mat4::identity();
	mat4 SPV_mvpMatrix = mat4::identity();
	mat4 SPV_rotateMatrix = mat4::identity();
	mat4 SPV_translateMatrix = mat4::identity();

	//code
	SPV_mvpMatrix = SPV_projectionMatrix * SPV_viewMatrix * SPV_modelMatrix;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(SPV_gShaderProgramObject);
	glUniformMatrix4fv(SPV_uProjectionMatrixUniform, 1, GL_FALSE, SPV_mvpMatrix);
	glUniform3fv(uColorUniform,1, color[colorIndex]);

	if(curGridSize != MY_ARRAY)
	{
		curGridSize = MY_ARRAY;
		pos = (float *)realloc(pos, MY_ARRAY * sizeof(float));

		if(devVertextArray) cudaFree(devVertextArray);

		auto cudaOp = cudaMalloc(&devVertextArray, sizeof(float)*MY_ARRAY);
		if(cudaOp != cudaError::cudaSuccess)
		{
			uninitialize();
			exit(1);
		}
	}
	
	if(bGPU)
	{
		launchGPUKernel <<< meshWidth, meshHeight >>> (meshWidth, meshHeight, animTime, devVertextArray);
		cudaMemcpy(pos, devVertextArray, MY_ARRAY * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}
	else
	{
		launchCPUKernel(meshWidth, meshHeight, animTime);
	}

	glBindBuffer(GL_ARRAY_BUFFER, SPV_vbo_pyramid_position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MY_ARRAY, pos, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(SPV_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SPV_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(SPV_vao_pyramid);
	glDrawArrays(GL_POINTS, 0, meshWidth * meshHeight);
	glBindVertexArray(0);
	
	glUseProgram(0);

	glXSwapBuffers(SPV_gpDisplay, SPV_gWindow);


	animTime = animTime + 0.02f;
	
	fps++;
	auto elapsedMSec = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startTime).count();
	if(elapsedMSec > 1000)
	{
		startTime = std::chrono::high_resolution_clock::now();
		gfps = fps;
		printf("Application FPS : %u \n", fps);
		fps = 0;
	}

}

void uninitialize(void)
{
	//local variables declarations 
	GLXContext SPV_currentGLXContext;

	//code
	//unuse current running program
	glUseProgram(0);
	
	if(devVertextArray) cudaFree(devVertextArray);

	if(pos) free(pos);

	//delete vao and vbo
	if(SPV_vbo_pyramid_color)
	{
		glDeleteBuffers(1, &SPV_vbo_pyramid_color);
		SPV_vbo_pyramid_color = 0;
	}

	if(SPV_vbo_pyramid_position)
	{
		glDeleteBuffers(1, &SPV_vbo_pyramid_position);
		SPV_vbo_pyramid_position = 0;
	}

	if(SPV_vao_pyramid)
	{
		glDeleteVertexArrays(1, &SPV_vao_pyramid);
		SPV_vao_pyramid = 0;
	}

	//all shader detach area
	glDetachShader(SPV_gShaderProgramObject, SPV_gVertexShaderObject);
	glDetachShader(SPV_gShaderProgramObject, SPV_gFragmentShaderObject);

	//all shader delete area
	glDeleteShader(SPV_gVertexShaderObject);
	SPV_gVertexShaderObject = 0;
	glDeleteShader(SPV_gFragmentShaderObject);
	SPV_gFragmentShaderObject = 0;

	//delete shader program area
	glDeleteProgram(SPV_gShaderProgramObject);
	SPV_gShaderProgramObject = 0;

	//free all variables and pointers holding memory
	SPV_currentGLXContext = glXGetCurrentContext();
	if(SPV_currentGLXContext == SPV_gGLXContext)
	{
		glXMakeCurrent(SPV_gpDisplay, 0, 0);
		glXDestroyContext(SPV_gpDisplay, SPV_gGLXContext);
	}

	if(SPV_gWindow)
		XDestroyWindow(SPV_gpDisplay, SPV_gWindow);

	if(SPV_gColorMap)
		XFreeColormap(SPV_gpDisplay, SPV_gColorMap);

	if(SPV_gpXVisualInfo)
	{
		free(SPV_gpXVisualInfo);
		SPV_gpXVisualInfo = NULL;
	}

	if(SPV_gpDisplay)
	{
		XCloseDisplay(SPV_gpDisplay);
		SPV_gpDisplay = NULL;
	}
}

int checkShaderCompilationStatus(GLuint shaderObject, const char *shaderInfo)
{
	//local variables declaration area
	GLint SPV_iCompileLogLength =0;
	GLint SPV_iCompileStatus = 0;
	char *SPV_szCompileLog = NULL;

	//code
	glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &SPV_iCompileStatus);
	if(SPV_iCompileStatus == GL_FALSE)
	{
		glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &SPV_iCompileLogLength);
		if(SPV_iCompileLogLength > 0)
		{
			SPV_szCompileLog = (char *)malloc(SPV_iCompileLogLength);
			if(SPV_szCompileLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(shaderObject, SPV_iCompileLogLength, &written, SPV_szCompileLog);
				fprintf(SPV_debug, "%s : %s\n", shaderInfo, SPV_szCompileLog);
				printf("%s : %s \n", shaderInfo, SPV_szCompileLog);
				free(SPV_szCompileLog);
				return(1);
			}
		}
	}

	return(0);
}


int checkShaderProgramLinkStatus(GLuint shaderProgram, const char *programInfo)
{
	//local variables declaration area
	GLint SPV_iLinkLogLength = 0;
	GLint SPV_iLinkStatus = 0;
	char *SPV_szLinkLog = NULL;

	//code
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &SPV_iLinkStatus);
	if(SPV_iLinkStatus == GL_FALSE)
	{
		glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &SPV_iLinkLogLength);
		if(SPV_iLinkLogLength > 0)
		{
			SPV_szLinkLog = (char *)malloc(SPV_iLinkLogLength);
			if(SPV_szLinkLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(shaderProgram, SPV_iLinkLogLength, &written, SPV_szLinkLog);
				fprintf(SPV_debug, "%s : %s\n", programInfo, SPV_szLinkLog);
				printf("%s : %s \n", programInfo, SPV_szLinkLog);
				free(SPV_szLinkLog);
				return(1);
			}
		}
	}

	return(0);
}
