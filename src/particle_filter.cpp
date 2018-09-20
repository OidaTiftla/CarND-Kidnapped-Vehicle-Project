/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	// Set standard deviations for x, y, and theta.
	auto std_x = std[0];
	auto std_y = std[1];
	auto std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std_x);

	// Create normal distributions for y and theta.
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

    this->num_particles = 1000;
    for (int i = 0; i < this->num_particles; ++i) {
        auto p = Particle();
        p.id = i;
        p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
        p.weight = 1;

        this->particles.push_back(p);
    }
    this->weights = vector<double>(this->num_particles, 1.0);
    this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	// Set standard deviations for x, y, and theta.
	auto std_x = std_pos[0];
	auto std_y = std_pos[1];
	auto std_theta = std_pos[2];

	// Create normal (Gaussian) distributions for x, y and theta.
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

    for (auto &p : this->particles) {
        // move without error
        p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
        p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        p.theta += yaw_rate * delta_t;
        // add error
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
        while (p.theta > M_PI) {
            p.theta -= M_PI * 2;
        }
        while (p.theta < -M_PI) {
            p.theta += M_PI * 2;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto &obs : observations) {
        auto dist_squared_min = std::numeric_limits<const float>::infinity();
        for (auto &pred : predicted) {
            auto dist_squared = (obs.x - pred.x) * (obs.x - pred.x) + (obs.y - pred.y) * (obs.y - pred.y);
            if (dist_squared < dist_squared_min) {
                dist_squared_min = dist_squared;
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (auto &p : this->particles) {
        std::vector<LandmarkObs> observations_in_map_coordinates;
        for (auto const &obs : observations) {
            LandmarkObs obs_in_map_coordinates;
            obs_in_map_coordinates.x = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
            obs_in_map_coordinates.y = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;
            observations_in_map_coordinates.push_back(obs_in_map_coordinates);
        }

        std::vector<LandmarkObs> predicted;
        for (auto const &lm : map_landmarks.landmark_list) {
            LandmarkObs pred;
            pred.id = lm.id_i;
            pred.x = lm.x_f;
            pred.y = lm.y_f;
            predicted.push_back(pred);
        }

        this->dataAssociation(predicted, observations_in_map_coordinates);

        p.weight = 1;
        for (auto const &obs : observations_in_map_coordinates) {
            const auto pred = std::find_if(predicted.begin(), predicted.end(), [&obs](const LandmarkObs &x) { return x.id == obs.id; });
            auto prob = normpdf(obs.x, obs.y, pred->x, pred->y, std_landmark[0], std_landmark[1]);
            p.weight *= prob;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // copy weights from the particles into the array
    for (int i = 0; i < this->num_particles; ++i) {
        this->weights[i] = this->particles[i].weight;
    }

	default_random_engine gen;

	// Create uniform distributions for particle index and beta.
    uniform_int_distribution<int> dist_particle_index(0, this->particles.size() - 1);
    uniform_int_distribution<int> dist_beta(0, *std::max_element(this->weights.begin(), this->weights.end()) * 2);

    // resampling
    std::vector<Particle> new_particles;
    auto particle_index = dist_particle_index(gen);
    auto beta = 0.0f;
    for (int i = 0; i < this->num_particles; ++i) {
        beta += dist_beta(gen);
        while (this->particles[particle_index].weight < beta) {
            beta -= this->particles[particle_index].weight;
            ++particle_index;
            particle_index %= this->num_particles;
        }
        Particle p;
        p.id = i;
        p.x = this->particles[particle_index].x;
        p.y = this->particles[particle_index].y;
        p.theta = this->particles[particle_index].theta;
        new_particles.push_back(p);
    }
    this->particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
