#include <gs_math.hpp>

#include <gs_renderer_backend.hpp>

int main(int argc, char *argv[])
{
    gs_renderer_backend backend;

    if(!backend.awake())
    {
        printf("could not awake backend !!!\n");
        return -1;
    }

    while (!backend.is_closed())
    {
        backend.frame_start();
        backend.frame_update();
        backend.frame_render();
        backend.frame_finish();
    }
    
    backend.finish();
    backend.quit();

    return 0;
}
