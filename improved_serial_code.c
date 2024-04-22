#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define NUM_PARTICLES 20000
#define NUM_TIMESTEPS 10
#define GRAV_CONST 0.001

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double mass;
} Particle;

void init(Particle *particles, int num_particles);
void update_positions(Particle *particles, int num_particles);
void calculate_forces(Particle *particles, int num_particles);
void calculate_centre_of_mass(Particle *particles, int num_particles, double *com);

int main() {
    Particle *particles = (Particle *)malloc(NUM_PARTICLES * sizeof(Particle));

    init(particles, NUM_PARTICLES);

    double com[3];
    struct timeval start, end;

    gettimeofday(&start, NULL);

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
        calculate_forces(particles, NUM_PARTICLES);
        update_positions(particles, NUM_PARTICLES);
        calculate_centre_of_mass(particles, NUM_PARTICLES, com);
        printf("Timestep %d: Centre of mass = (%.3f, %.3f, %.3f)\n", t + 1, com[0], com[1], com[2]);
    }

    gettimeofday(&end, NULL);
    double exec_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1e6;

    printf("Execution time: %.6f seconds\n", exec_time);

    free(particles);

    return 0;
}

void init(Particle *particles, int num_particles) {
    #pragma omp parallel for
    for (int i = 0; i < num_particles; i++) {
        particles[i].x = -50.0 + 100.0 * (double)rand() / RAND_MAX;
        particles[i].y = -50.0 + 100.0 * (double)rand() / RAND_MAX;
        particles[i].z = 100.0 * (double)rand() / RAND_MAX;
        particles[i].vx = -5.0 + 10.0 * (double)rand() / RAND_MAX;
        particles[i].vy = -5.0 + 10.0 * (double)rand() / RAND_MAX;
        particles[i].vz = -5.0 + 10.0 * (double)rand() / RAND_MAX;
        particles[i].mass = 0.1 + 10.0 * (double)rand() / RAND_MAX;
    }
}

void calculate_forces(Particle *particles, int num_particles) {
    #pragma omp parallel for
    for (int i = 0; i < num_particles; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < num_particles; j++) {
            if (j != i) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;
                double d = sqrt(dx * dx + dy * dy + dz * dz);
                double F = GRAV_CONST * particles[i].mass * particles[j].mass / (d * d);
                ax += (F / particles[i].mass) * dx / d;
                ay += (F / particles[i].mass) * dy / d;
                az += (F / particles[i].mass) * dz / d;
            }
        }
        particles[i].vx += ax;
        particles[i].vy += ay;
        particles[i].vz += az;
    }
}

void update_positions(Particle *particles, int num_particles) {
    #pragma omp parallel for
    for (int i = 0; i < num_particles; i++) {
        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;
        particles[i].z += particles[i].vz;
    }
}

void calculate_centre_of_mass(Particle *particles, int num_particles, double *com) {
    double total_mass = 0.0;
    double com_x = 0.0, com_y = 0.0, com_z = 0.0;

    #pragma omp parallel for reduction(+:total_mass,com_x,com_y,com_z)
    for (int i = 0; i < num_particles; i++) {
        total_mass += particles[i].mass;
        com_x += particles[i].mass * particles[i].x;
        com_y += particles[i].mass * particles[i].y;
        com_z += particles[i].mass * particles[i].z;
    }

    com[0] = com_x / total_mass;
    com[1] = com_y / total_mass;
    com[2] = com_z / total_mass;
}
