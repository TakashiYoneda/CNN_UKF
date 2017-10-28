#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;


  ///* time when the state is true, in us
  previous_timestamp_ = 0;


  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter (originl=3)
  lambda_ = 3 - n_aug_;                                                // <=== big influence to change


  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);


  // The values for the process noise std_a_ and std_yawdd_ are both set to 30.
  // These will need to be adjusted in order to get your Kalman filter working.
  // Think about what a standard deviation of 30 means.
  // For a Gaussian distribution, we expect the acceleration to be between
  //   −60m/s^2​​ and +60m/s^2​
  //   −60rad/s^2​​ and +60rad/s^2​​
  // ninety-five percent of the time . That seems quite high!
  //   To put those values in perspective,
  // the fastest measured linear acceleration for a street legal sports car is
  // currently 0 to 60 mph in 2.2 seconds.
  // 0 to 60 mph in 2.2 seconds is about 12m/s^2.
  // The bike simulation probably tends to have even lower acceleration.

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;


  ///* Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);


  //
  // Measurement Noise below should be kept in original value
  //
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // NISs kai(2) 0.05 = 6.0, kai(3) 0.05 = 7.8
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "UKF: " << endl;
    x_ = VectorXd(n_x_);
    x_ << 1, 1, 1, 1, 1;


    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      //vector for conversion ( rho, phi -> px, py )
      //px = x_state[0], py = x_state[1]
      VectorXd x_state(2);
      x_state(0) = meas_package.raw_measurements_[0] * cos (meas_package.raw_measurements_[1]);
      x_state(1) = meas_package.raw_measurements_[0] * sin (meas_package.raw_measurements_[1]);
      x_ << x_state(0), x_state(1), 0, 0, 0;
  		previous_timestamp_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
  		previous_timestamp_ = meas_package.timestamp_;
    }

//
    P_ << 0.1, 0, 0, 0, 0,
          0, 0.1, 0, 0, 0,
          0, 0, 2, 0, 0,
          0, 0, 0, 0.01, 0,
          0, 0, 0, 0, 0.01;

/*
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
*/
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }


  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;


  Prediction(dt);

  cout<<x_<<endl;


  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);

  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);

  }




}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //======================================================
  // Generate Sigma points with augmentation
  //======================================================

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);


  //create augmented mean state
  x_aug.head(n_x_) = x_;    // 5 (0~4)
  x_aug(n_x_) = 0;          // 5
  x_aug(n_x_ +1) = 0;        // 6

  //create augmented covariance matrix
  //   P k|k (5x5)       0             0
  //     0           Q(nu_a^2)         0
  //     0               0        Q(nu_phi_dd^2)
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_*std_a_;
  P_aug(n_x_ +1, n_x_ +1) = std_yawdd_*std_yawdd_;

  //create square root of augmented covariance matrix
  MatrixXd L = P_aug.llt().matrixL();

  //Generate augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // output: Xsig_aug(7x7)


/*
  //print result
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;
*/


  //======================================================
  // Predict Sigma points with augmentation (7dim => 5dim)
  //======================================================

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // output: Xsig_pred(5x)

/*
  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;
*/


//======================================================
// Predict Mean & Covariance matrix of the Predicted State
//======================================================

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }


  //create tempo state vector for prediction
  VectorXd x = VectorXd(n_x_);

  //create tempo covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
     x = x + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }


  // output: predicted x_, predicted P_
  x_ = x;
  P_ = P;

/*
  //print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P << std::endl;
*/

}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    //======================================================
    // Transform to measurement space
    //======================================================

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

     // measurement model
     Zsig(0,i) = Xsig_pred_(0,i);                        //px
     Zsig(1,i) = Xsig_pred_(1,i);                        //py
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
       z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
     //residual
     VectorXd z_diff = Zsig.col(i) - z_pred;

     S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_*std_laspx_, 0,
           0, std_laspy_*std_laspy_;
    S = S + R;

    // output: mean z_pred, covariance S

  /*
    //print result
    std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    std::cout << "S: " << std::endl << S << std::endl;

    //write result
    *z_out = z_pred;
    *S_out = S;
  */


  //======================================================
  // Update state
  //======================================================

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

      //residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // z define
    VectorXd z = VectorXd(n_z);
    z(0) = meas_package.raw_measurements_(0);
    z(1) = meas_package.raw_measurements_(1);

    //residual
    VectorXd z_diff = z - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

  /*
    //print result
    std::cout << "Updated state x: " << std::endl << x << std::endl;
    std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

    //write result
    *x_out = x;
    *P_out = P;
  */

  //======================================================
  // Calculate NIS
  //======================================================

    //calculate epsiron
    NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;


}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  **/

  //======================================================
  // Transform to measurement space
  //======================================================

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

   // extract values for better readibility
   double p_x = Xsig_pred_(0,i);
   double p_y = Xsig_pred_(1,i);
   double v  = Xsig_pred_(2,i);
   double yaw = Xsig_pred_(3,i);

   double v1 = cos(yaw)*v;
   double v2 = sin(yaw)*v;

   // measurement model
   Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
   Zsig(1,i) = atan2(p_y,p_x);                                 //phi
   Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
     z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
   //residual
   VectorXd z_diff = Zsig.col(i) - z_pred;

   //angle normalization
   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

   S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
         0, std_radphi_*std_radphi_, 0,
         0, 0,std_radrd_*std_radrd_;
  S = S + R;

  // output: mean z_pred, covariance S

/*
  //print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *z_out = z_pred;
  *S_out = S;
*/


//======================================================
// Update state
//======================================================

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // z define
  VectorXd z = VectorXd(n_z);
  z(0) = meas_package.raw_measurements_(0);
  z(1) = meas_package.raw_measurements_(1);
  z(2) = meas_package.raw_measurements_(2);

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

/*
  //print result
  std::cout << "Updated state x: " << std::endl << x << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

  //write result
  *x_out = x;
  *P_out = P;
*/

//======================================================
// Calculate NIS
//======================================================

  //calculate epsiron
  NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;


}
