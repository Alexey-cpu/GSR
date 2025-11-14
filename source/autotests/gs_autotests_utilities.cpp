#include <gs_math.hpp>

int main(int argc, char *argv[])
{
    printf("running %s\n", "gs_autotests_utilities.cpp");

    // gs_max
    GS_ASSERT(gs_max(10, 12, 3, 0, 22, -1) == 22);
    printf("gs_max success...\n");

    // gs_min
    GS_ASSERT(gs_min(10, 12, 3, 0, 22, -1) == -1);
    printf("gs_min success... \n");

    GS_ASSERT(gs_sum_of_squares(2, 3, 4, 5) == (2 * 2 + 3 * 3 + 4 * 4 + 5 * 5));
    printf("gs_sum_of_squares success... \n");

    GS_ASSERT(gs_vector_length (2, 3, 4, 5) == sqrt(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5));
    printf("gs_vector_length success... \n");

    printf("\n");

    return 0;
}