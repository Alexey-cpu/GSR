#include <gs_renderer_backend.hpp>

// GLAD
#include <glad/glad.h> 

// STB
#include <stb_image.h>
#include <stb_image_write.h>

// GLFW
#include <GLFW/glfw3.h>

// Callbacks
void OpenGLPlatformBackendOnWindowResize(GLFWwindow* _Window, int _Width, int _Height)
{
    (void)_Window;
    glViewport(0, 0, _Width, _Height);
}

void OpenGLPlatformBackendOnWindowMaximizedCallback(GLFWwindow* _Window, int _Maximized)
{
    int width  = 0;
    int height = 0;
    glfwGetWindowSize(_Window, &width, &height);
    glViewport(0, 0, width, height);
}

gs_renderer_backend::gs_renderer_backend(){}
gs_renderer_backend::~gs_renderer_backend(){}

bool gs_renderer_backend::awake(const char* _Name, const gs_renderer_context_hints& _Hints, void* _Share)
{
    // initialization
    if(glfwInit() == GLFW_FALSE)
        return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    if(_Hints & gs_renderer_context_hints_::gs_renderer_context_hints_visible)
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    else
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    if(_Hints & gs_renderer_context_hints_::gs_renderer_context_hints_decorated)
        glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    else
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

    if(_Hints & gs_renderer_context_hints_::gs_renderer_context_hints_resizable)
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    else
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    if(_Hints & gs_renderer_context_hints_::gs_renderer_context_hints_iconified)
        glfwWindowHint(GLFW_ICONIFIED, GLFW_TRUE);
    else
        glfwWindowHint(GLFW_ICONIFIED, GLFW_FALSE);

    if(_Hints & gs_renderer_context_hints_::gs_renderer_context_hints_focused)
        glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);
    else
        glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);

    // create context
    Context = glfwCreateWindow(
        512,
        256,
        _Name,
        nullptr,
        reinterpret_cast<GLFWwindow*>(_Share));

    if(Context == nullptr)
    {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(reinterpret_cast<GLFWwindow*>(Context));

    // setup callbacks
    glfwSetWindowSizeCallback(reinterpret_cast<GLFWwindow*>(Context), &OpenGLPlatformBackendOnWindowResize);
    glfwSetFramebufferSizeCallback(reinterpret_cast<GLFWwindow*>(Context), &OpenGLPlatformBackendOnWindowResize);
    glfwSetWindowMaximizeCallback(reinterpret_cast<GLFWwindow*>(Context), &OpenGLPlatformBackendOnWindowMaximizedCallback);

    // load OpenGL interface using GLAD
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwTerminate();
        return false;
    }

    // maximize context window
    glfwMaximizeWindow(reinterpret_cast<GLFWwindow*>(Context));

    // call window maximize callback if the Window has been maximized
    OpenGLPlatformBackendOnWindowMaximizedCallback(
        reinterpret_cast<GLFWwindow*>(Context),
        glfwGetWindowAttrib(reinterpret_cast<GLFWwindow*>(Context), GLFW_MAXIMIZED));

    return true;
}

void gs_renderer_backend::frame_start()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glClear(GL_STENCIL_BUFFER_BIT);
    glClearColor(1.f, 1.f, 1.f, 1.f);
    glfwPollEvents();
    glfwSwapInterval(1);
}

void gs_renderer_backend::frame_update()
{
    // stretch viewport to cover all the context window
    int display_w = 0;
    int display_h = 0;
    glfwGetFramebufferSize(reinterpret_cast<GLFWwindow*>(Context), &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
}

void gs_renderer_backend::frame_render()
{
}

void gs_renderer_backend::frame_finish()
{
    glfwSwapBuffers(reinterpret_cast<GLFWwindow*>(Context));
}

void gs_renderer_backend::finish()
{
}

void gs_renderer_backend::quit()
{
    glfwDestroyWindow(reinterpret_cast<GLFWwindow*>(Context));
    glfwTerminate();
    Context = nullptr;
}

bool gs_renderer_backend::is_closed() const
{
    return glfwWindowShouldClose(reinterpret_cast<GLFWwindow*>(Context));
}

void gs_renderer_backend::close()
{
    glfwSetWindowShouldClose(reinterpret_cast<GLFWwindow*>(Context), GL_TRUE);
}

gs_renderer_texture gs_renderer_backend_construct_image(
    const gs_renderer_backend&            _Backend,
    const unsigned char*                  _RawBuffer,
    const int&                            _Width,
    const int&                            _Height,
    const gs_renderer_texture_format&     _Format,
    const gs_renderer_texture_wrap_mode&  _Wrap,
    const gs_renderer_texture_min_filter& _MinFilter,
    const gs_renderer_texture_max_filter& _MaxFilter)
{
    return gs_renderer_texture();
}

gs_renderer_texture gs_renderer_backend_construct_image(
    const gs_renderer_backend&            _Backend,
    const char*                           _FilePath,
    const gs_renderer_texture_format&     _Format,
    const gs_renderer_texture_wrap_mode&  _Wrap,
    const gs_renderer_texture_min_filter& _MinFilter, 
    const gs_renderer_texture_max_filter& _MaxFilter)
{
    return gs_renderer_texture();
}

void destroy_image(const gs_renderer_texture& _Texture)
{
}