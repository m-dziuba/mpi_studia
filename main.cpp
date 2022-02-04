#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <mpi.h>


void generate_random_vector(int N, float *vec, float lower_limit, float higher_limit);
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

    add(rank, size, static_cast<int>(pow(2, 29)));
    get_norm(rank, size, static_cast<int>(pow(2, 30)));
    get_histogram(rank, size, static_cast<int>(pow(2, 31)), 100);

    MPI_Finalize();
}


void generate_random_vector(int N, float *vec, float lower_limit, float higher_limit) {
    std::random_device rd;
    std::mt19937_64 eng(rd());

    std::uniform_real_distribution<float> distribution(lower_limit,higher_limit);
    for (int i = 0; i < N; i++) {
        vec[i] = distribution(eng);
    }
}

void generate_random_vector(int N, int *vec, int lower_limit, int higher_limit) {
    std::random_device rd;
    std::mt19937_64 eng(rd());

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
        out_file << numbers[i] << " ";
    }

    out_file.close();
    return EXIT_SUCCESS;
}

void add(int rank, int size, int N) {
    double start, stop;

    int chunk = N / size;

    auto *vec_a = new float[N];
    auto *vec_b = new float[N];
    auto *vec_c = new float[N];

    auto *chunk_vec_a = new float[chunk];
    auto *chunk_vec_b = new float[chunk];
    auto *chunk_vec_c = new float[chunk];

    if (rank == 0) {
        generate_random_vector(N, vec_b, 0, 1.0);
        generate_random_vector(N, vec_c, 0, 1.0);
    }

    start = MPI_Wtime();

    MPI_Scatter(vec_b, chunk, MPI_FLOAT,
                chunk_vec_b, chunk, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vec_c, chunk, MPI_FLOAT,
                chunk_vec_c, chunk, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        chunk_vec_a[i] = chunk_vec_b[i] + chunk_vec_c[i];
    }

    MPI_Gather(chunk_vec_a, chunk, MPI_FLOAT,
               vec_a, chunk, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    stop = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n================================" << std::endl;
        std::cout << "Zad. E1a Dodawanie" << std::endl;
        std::cout << "Processes used: " << size << std::endl;
        std::cout << "Time: " << stop - start << std::endl;
        std::cout << "A[0]: " << vec_a[0] <<", A[N - 1]: " << vec_a[N - 1]<< std::endl;
    }

    delete[] vec_a;
    delete[] vec_b;
    delete[] vec_c;
    delete[] chunk_vec_a;
    delete[] chunk_vec_b;
    delete[] chunk_vec_c;
}

void get_norm(int rank, int size, int N) {
    double start, stop;

    int chunk{N / size};
    auto *vec_a = new float[N];
    auto *chunk_vec_a = new float[chunk];
    double result{};

    if (rank == 0) {
        generate_random_vector(N, vec_a, 0, 1);
    }

    start = MPI_Wtime();

    MPI_Scatter(vec_a, chunk, MPI_FLOAT,
                chunk_vec_a, chunk, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        chunk_vec_a[i] *= chunk_vec_a[i];
    }

    MPI_Gather(chunk_vec_a, chunk, MPI_FLOAT,
               vec_a, chunk, MPI_FLOAT,
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

    delete[] vec_a;
    delete[] chunk_vec_a;
}

void get_histogram(int rank, int size, int N, int range) {
    int num;
    double start, stop;

    range++;

    int chunk{N / size};
    auto *histogram = new int[range]{};
    auto *chunk_histogram = new int[range]{0};
    auto *numbers = new int[N];
    auto *chunk_numbers = new int[chunk];

    if (rank == 0) {
        // z jakiegoś powodu inaczej nie mogłem otworzyć pliku
        std::string filename{"/home/mateusz/mpi_studia/numbers.txt"};
        generate_file(N, range, filename);
        std::ifstream in_file{filename};
        int i{0};
        while (in_file >> num && i < N) {
            numbers[i] = num;
            i++;
        }
    }

    start = MPI_Wtime();

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
        std::cout << std::setw(8) << std::left << "Number"
                  << std::setw(9) << std::right << "Frequency | "
                  << std::setw(8) << std::left << "Number"
                  << std::setw(9) << std::right << "Frequency | "
                  << std::setw(8) << std::left << "Number"
                  << std::setw(9) << std::right << "Frequency |"
                  << std::endl;
        for (int i = 0; i < 35; i++) {
            std::cout << std::setw(8) << std::left << i
                      << std::setw(9) << std::right << histogram[i] << " | "
                      << std::right << std::setw(8) << std::left << i + 34
                      << std::setw(9) << std::right << histogram[i + 34] << " | ";
            if (i != 34) {
                std::cout << std::right << std::setw(8) << std::left << i + 67
                          << std::setw(9) << std::right << histogram[i + 67] << " | "
                          << std::right <<std::endl;
            } else {
                std::cout << std::endl;
            }
         }
    }
    delete[] histogram;
    delete[] chunk_histogram;
    delete[] numbers;
    delete[] chunk_numbers;
}