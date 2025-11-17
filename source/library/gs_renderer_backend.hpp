#pragma once

#include <gs_math.hpp>

// Hints
enum gs_renderer_context_hints_ : int
{
    // tweaks
    gs_renderer_context_hints_none,
    gs_renderer_context_hints_visible,
    gs_renderer_context_hints_decorated,
    gs_renderer_context_hints_resizable,
    gs_renderer_context_hints_iconified,
    gs_renderer_context_hints_focused,
    gs_renderer_context_hints_default =
        gs_renderer_context_hints_visible   |
        gs_renderer_context_hints_decorated |
        gs_renderer_context_hints_resizable |
        gs_renderer_context_hints_focused
};

enum gs_renderer_texture_format_ : int
{
    gs_renderer_texture_format_alpha,
    gs_renderer_texture_format_rgb,
    gs_renderer_texture_format_rgba,
};

enum gs_renderer_texture_wrap_mode_ : int
{
    gs_renderer_texture_wrap_mode_repeat,
    gs_renderer_texture_wrap_mode_mirrored,
    gs_renderer_texture_wrap_mode_clamp_to_edge,
    gs_renderer_texture_wrap_mode_clamp_to_border
};

enum gs_renderer_texture_min_filter_ : int
{
    gs_renderer_texture_min_filter_linear,
    gs_renderer_texture_min_filter_nearest,
    gs_renderer_texture_min_filter_nearest_mip_map_linear,
    gs_renderer_texture_min_filter_nearest_mip_map_nearest,
    gs_renderer_texture_min_filter_linear_mip_map_linear,
    gs_renderer_texture_min_filter_linear_mip_map_nearest,
};

enum gs_renderer_texture_max_filter_ : int
{
    gs_renderer_texture_max_filter_linear,
    gs_renderer_texture_max_filter_nearest,
};

typedef int gs_renderer_context_hints;
typedef int gs_renderer_texture_format;
typedef int gs_renderer_texture_wrap_mode;
typedef int gs_renderer_texture_min_filter;
typedef int gs_renderer_texture_max_filter;

// Enities
struct gs_renderer_texture final
{
    gs_renderer_texture(const gs_renderer_texture& _Other) :
        Ptr(_Other.Ptr),
        Width(_Other.Width),
        Height(_Other.Height),
        Format(_Other.Format),
        Wrap(_Other.Wrap),
        MinFilter(_Other.MinFilter),
        MaxFilter(_Other.MaxFilter){}

    gs_renderer_texture(
        const unsigned int&                   _Ptr       = 0,
        const int&                            _Width     = 128,
        const int&                            _Height    = 128,
        const gs_renderer_texture_format&     _Format    = gs_renderer_texture_format_rgba,
        const gs_renderer_texture_wrap_mode&  _Wrap      = gs_renderer_texture_wrap_mode_::gs_renderer_texture_wrap_mode_repeat,
        const gs_renderer_texture_min_filter& _MinFilter = gs_renderer_texture_min_filter_::gs_renderer_texture_min_filter_linear,
        const gs_renderer_texture_max_filter& _MaxFilter = gs_renderer_texture_max_filter_::gs_renderer_texture_max_filter_linear) : 
    Width(_Width),
    Height(_Height),
    Ptr(_Ptr),
    Format(_Format),
    Wrap(_Wrap),
    MinFilter(_MinFilter),
    MaxFilter(_MaxFilter){}

    const unsigned int                   Ptr       {+0};
    const int                            Width     {-1};
    const int                            Height    {-1};
    const gs_renderer_texture_format     Format    {gs_renderer_texture_format_::gs_renderer_texture_format_rgba};
    const gs_renderer_texture_wrap_mode  Wrap      {gs_renderer_texture_wrap_mode_::gs_renderer_texture_wrap_mode_repeat};
    const gs_renderer_texture_min_filter MinFilter {gs_renderer_texture_min_filter_::gs_renderer_texture_min_filter_linear};
    const gs_renderer_texture_max_filter MaxFilter {gs_renderer_texture_max_filter_::gs_renderer_texture_max_filter_linear};
};

struct gs_renderer_shader final
{
    gs_renderer_shader(const gs_renderer_shader& _Other) : Ptr(_Other.Ptr){}
    gs_renderer_shader(const unsigned int& _Ptr = 0) : Ptr(_Ptr){}

    const unsigned int Ptr {0};
};

struct gs_renderer_vertex final
{
    gs_renderer_vertex(const gs_renderer_vertex& _Other) :
        Position(_Other.Position),
        Normal(_Other.Normal),
        UV(_Other.UV){}

    gs_renderer_vertex(
        const gs_vec3f& _Position = gs_vec3f(0),
        const gs_vec3f& _Normal   = gs_vec3f(0),
        const gs_vec2f& _UV       = gs_vec2f(0)) :
    Position(_Position),
    Normal(_Normal),
    UV(_UV){}

    gs_vec3f Position;
    gs_vec3f Normal;
    gs_vec2f UV;
};

struct gs_renderer_mesh final
{
    gs_renderer_mesh(const gs_renderer_mesh& _Other) :
        VBO(_Other.VBO),
        VAO(_Other.VAO),
        EBO(_Other.EBO){}

    gs_renderer_mesh(
        const unsigned int& _VBO = 0,
        const unsigned int& _VAO = 0,
        const unsigned int& _EBO = 0) :
    VBO(_VBO),
    VAO(_VAO),
    EBO(_EBO){}

    const unsigned int VBO{0};
    const unsigned int VAO{0};
    const unsigned int EBO{0};
};

class gs_renderer_backend final
{
public:
    gs_renderer_backend();
    ~gs_renderer_backend();

    bool awake(const char* _Name = "GS_Renderer", const gs_renderer_context_hints& _Hints = gs_renderer_context_hints_default, void* _Share = nullptr);
    void frame_start();
    void frame_update();
    void frame_render();
    void frame_finish();
    void finish();
    void quit();
    bool is_closed() const;
    void close();

protected:
    void* Context{nullptr};
};

// API
gs_renderer_texture gs_renderer_backend_construct_image(
    const gs_renderer_backend&            _Backend,
    const unsigned char*                  _RawBuffer,
    const int&                            _Width,
    const int&                            _Height,
    const gs_renderer_texture_format&     _Format    = gs_renderer_texture_format_::gs_renderer_texture_format_rgba,
    const gs_renderer_texture_wrap_mode&  _Wrap      = gs_renderer_texture_wrap_mode_::gs_renderer_texture_wrap_mode_repeat,
    const gs_renderer_texture_min_filter& _MinFilter = gs_renderer_texture_min_filter_::gs_renderer_texture_min_filter_linear, 
    const gs_renderer_texture_max_filter& _MaxFilter = gs_renderer_texture_max_filter_::gs_renderer_texture_max_filter_linear);

gs_renderer_texture gs_renderer_backend_construct_image(
    const gs_renderer_backend&            _Backend,
    const char*                           _FilePath,
    const gs_renderer_texture_format&     _Format    = gs_renderer_texture_format_::gs_renderer_texture_format_rgba,
    const gs_renderer_texture_wrap_mode&  _Wrap      = gs_renderer_texture_wrap_mode_::gs_renderer_texture_wrap_mode_repeat,
    const gs_renderer_texture_min_filter& _MinFilter = gs_renderer_texture_min_filter_::gs_renderer_texture_min_filter_linear, 
    const gs_renderer_texture_max_filter& _MaxFilter = gs_renderer_texture_max_filter_::gs_renderer_texture_max_filter_linear);

void destroy_image(const gs_renderer_texture& _Texture);