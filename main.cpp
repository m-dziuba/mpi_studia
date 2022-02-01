#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <mpi.h>


void generate_random_vector(int N, double *vec, int lower_limit, int higher_limit);
void generate_random_vector(int N, int *vec, int lower_limit, int higher_limit);
int generate_file(int N, int range, std::string &filename);
void add(int rank, int size, int N);
void get_norm(int rank, int size, int N);
void get_histogram(int rank, int size, int N, int range);

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    add(rank, size, 1048576);
    get_norm(rank, size, 1024);
    get_histogram(rank, size, 1024, 100);

    MPI_Finalize();
}


void generate_random_vector(int N, double *vec, int lower_limit, int higher_limit) {
    std::random_device rd;
    std::default_random_engine eng(rd());

    std::uniform_real_distribution<double> distribution(lower_limit, higher_limit);

    for (int i = 0; i < N; i++) {
        vec[i] = distribution(eng);
    }
}

void generate_random_vector(int N, int *vec, int lower_limit, int higher_limit) {
    std::random_device rd;
    std::default_random_engine eng(rd());

    std::uniform_int_distribution<int> distribution(lower_limit, higher_limit);

    for (int i = 0; i < N; i++) {
        vec[i] = distribution(eng);
    }
}

int generate_file(int N, int range, std::string &filename) {
    std::ofstream out_file{filename, std::ios::trunc};

    if (!out_file) {
        std::cerr << "File was not opened" << std::endl;
        return EXIT_FAILURE;
    }

    auto *numbers = new int[N];

    generate_random_vector(N, numbers, 0 , range);

    for (int i = 0; i < N; i++) {
        out_file << numbers[i] << std::endl;
    }

    out_file.close();
    return EXIT_SUCCESS;
}

void add(int rank, int size, int N) {
    double start, stop;

    int chunk = N / size;

    auto *vec_a = new double[N];
    auto *vec_b = new double[N];
    auto *vec_c = new double[N];

    auto *chunk_vec_a = new double[chunk];
    auto *chunk_vec_b = new double[chunk];
    auto *chunk_vec_c = new double[chunk];

    start = MPI_Wtime();

    if (rank == 0) {
        generate_random_vector(N, vec_b, 0, 1);
        generate_random_vector(N, vec_c, 0, 1);
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
        std::cout << "Zad. E1a Dodawanie" << std::endl;
        std::cout << "Processes used: " << size << std::endl;
        std::cout << "Time: " << stop - start << std::endl;
        std::cout << "A[0]: " << vec_a[0] <<", A[N - 1]: " << vec_a[N - 1]<< std::endl;
    }
}

void get_norm(int rank, int size, int N) {
    double start, stop;

    start = MPI_Wtime();

    int chunk{N / size};
    auto *vec_a = new double[N];
    auto *chunk_vec_a = new double[chunk];
    double result{0};

    if (rank == 0) {
        generate_random_vector(N, vec_a, 0, 1);
    }

    MPI_Scatter(vec_a, chunk, MPI_DOUBLE,
                chunk_vec_a, chunk, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        chunk_vec_a[i] *= chunk_vec_a[i];
    }

    MPI_Gather(chunk_vec_a, chunk, MPI_DOUBLE,
               vec_a, chunk, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            result += vec_a[i];
        }
        result = sqrt(result);
        std::cout << "\n================================" << std::endl;
        std::cout << "Zad. E1b Norma" << std::endl;
        std::cout << "Processes used: " << size << std::endl;
        std::cout << "Time: " << stop - start << std::endl;
        std::cout << "Norm(A): " << result << std::endl;
    }
}

void get_histogram(int rank, int size, int N, int range) {
    int num;
    double start, stop;

    range++;
    start = MPI_Wtime();

    int chunk{N / size};
    auto *histogram = new int[range]{};
    auto *chunk_histogram = new int[range]{0};
    auto *numbers = new int[N];
    auto *chunk_numbers = new int[chunk];

    if (rank == 0) {
        std::string filename{"/home/mateusz/mpi_studia/numbers.txt"};
        generate_file(N, range, filename);
        std::ifstream in_file{filename};
        int i{0};
        while (in_file >> num) {
            numbers[i] = num;
            i++;
        }
    }

    MPI_Scatter(numbers, chunk, MPI_INT,
                chunk_numbers, chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        int loc = chunk_numbers[i];
        chunk_histogram[loc] += 1;
    }

    MPI_Reduce(chunk_histogram, histogram, range,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n================================" << std::endl;
        std::cout << "Zad. E1c Histogram" << std::endl;
        std::cout << "Processes used: " << size << std::endl;
        std::cout << "Time: " << stop - start << std::endl;
        std::cout << std::setw(10) << std::left << "Number" << "Frequency"
        << std::endl;
        for (int i = 0; i < range; i++) {
            std::cout << std::setw(10) << std::left << i << histogram[i]
                      << std::endl;
         }
    }
}