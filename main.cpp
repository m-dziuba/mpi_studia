#include <iostream>
#include <random>
#include <mpi.h>

void add(int argc, char *argv[]);
void get_norm(int argc, char *argv[]);
double get_histogram();
void generate_random_vector(int N, double *vec);

int main(int argc, char *argv[]) {
//    add(argc, argv);
    get_norm(argc, argv);
}

void generate_random_vector(int N, double *vec) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int i = 0; i < N; i++) {
        vec[i] = distribution(eng);
    }
}

void add(int argc, char *argv[]) {
    int rank, size;
    int N = 1048576;
    double start, stop;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;

    auto *vec_a = new double[N];
    auto *vec_b = new double[N];
    auto *vec_c = new double[N];

    auto *chunk_vec_a = new double[chunk];
    auto *chunk_vec_b = new double[chunk];
    auto *chunk_vec_c = new double[chunk];

    start = MPI_Wtime();

    if (rank == 0) {
        generate_random_vector(N, vec_b);
        generate_random_vector(N, vec_c);
    }

    MPI_Scatter(vec_b, chunk, MPI_DOUBLE,
                chunk_vec_b, chunk, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vec_c, chunk, MPI_DOUBLE,
                chunk_vec_c, chunk, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        chunk_vec_a[i] = chunk_vec_b[i] + chunk_vec_c[i];
    }

    MPI_Gather(chunk_vec_a, chunk, MPI_DOUBLE,
               vec_a, chunk, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n================================" << std::endl;
        std::cout << "Processes used: " << size << std::endl;
        std::cout << "Time: " << stop - start << std::endl;
        std::cout << "A[0]: " << vec_a[0] <<", A[N - 1]: " << vec_a[N - 1]<< std::endl;
    }

    MPI_Finalize();
}

void get_norm(int argc, char *argv[]) {
    int rank, size;
    int N{1024};
    double start{}, stop{};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start = MPI_Wtime();

    int chunk{size/N};
    auto *vec_a = new double[N];
    auto *chunk_vec_a = new double[chunk];

    if (rank == 0) {
        generate_random_vector(N, vec_a);
    }

    MPI_Scatter(vec_a, chunk, MPI_DOUBLE,
                chunk_vec_a, chunk, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        chunk_vec_a[i] *= chunk_vec_a[i];
    }

    MPI_Reduce(chunk_vec_a, vec_a, chunk,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    stop = MPI_Wtime();
    MPI_Finalize();
}