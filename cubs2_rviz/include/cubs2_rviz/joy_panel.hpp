// Copyright 2025 CogniPilot Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <cubs2_msgs/msg/aircraft_control.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/float32.hpp>

namespace cubs2
{

// Custom 2D joystick widget for aileron/elevator control
class JoystickWidget : public QWidget {
  Q_OBJECT

public:
  explicit JoystickWidget(QWidget * parent = nullptr);
  QSize sizeHint() const override {return QSize(200, 200);}

  double getAileron() const {return aileron_;}      // -1 to 1
  double getElevator() const {return elevator_;}    // -1 to 1

  void setTrim(double ail_trim, double elev_trim)
  {
    aileron_trim_ = ail_trim;
    elevator_trim_ = elev_trim;
    update();
  }

  void reset()
  {
    aileron_ = 0.0;
    elevator_ = 0.0;
    aileron_trim_ = 0.0;
    elevator_trim_ = 0.0;
    update();
  }

  void setPosition(double aileron, double elevator)
  {
    aileron_ = aileron;
    elevator_ = elevator;
    update();
    repaint();  // Force immediate repaint
  }

Q_SIGNALS:
  void joystickMoved(double aileron, double elevator);

protected:
  void paintEvent(QPaintEvent * event) override;
  void mousePressEvent(QMouseEvent * event) override;
  void mouseMoveEvent(QMouseEvent * event) override;
  void mouseReleaseEvent(QMouseEvent * event) override;

private Q_SLOTS:
  void springBackStep();

private:
  void updatePosition(const QPoint & pos);
  void startSpringBack();

  double aileron_{0.0};   // -1 (left) to +1 (right)
  double elevator_{0.0};  // -1 (down) to +1 (up)
  double aileron_trim_{0.0};
  double elevator_trim_{0.0};
  bool dragging_{false};
  QTimer * spring_timer_{nullptr};
};

class JoyPanel : public rviz_common::Panel {
  Q_OBJECT

public:
  explicit JoyPanel(QWidget * parent = nullptr);
  ~JoyPanel() override;

private Q_SLOTS:
  void onJoystickMoved(double aileron, double elevator);
  void onThrottleChanged(int value);
  void onRudderChanged(int value);
  void onAileronTrimChanged(int value);
  void onElevatorTrimChanged(int value);
  void publishControlInputs();
  void onResetClicked();
  void onEnabledChanged(int state);
  void onOnboardModeChanged(int index);
  void onInputModeChanged(int index);

private:
  void controlCallback(const cubs2_msgs::msg::AircraftControl::SharedPtr msg);
  void updateDisplayFromExternal(double aileron, double elevator, double throttle, double rudder);

private:
  // Virtual joystick controls
  JoystickWidget * joystick_{nullptr};
  QSlider * throttle_slider_{nullptr};
  QSlider * rudder_slider_{nullptr};
  QSlider * aileron_trim_slider_{nullptr};
  QSlider * elevator_trim_slider_{nullptr};
  QLabel * throttle_label_{nullptr};
  QLabel * rudder_label_{nullptr};
  QLabel * aileron_trim_label_{nullptr};
  QLabel * elevator_trim_label_{nullptr};
  QTimer * control_timer_{nullptr};
  QTimer * ros_spin_timer_{nullptr};
  QPushButton * reset_button_{nullptr};
  QCheckBox * enable_checkbox_{nullptr};
  QComboBox * onboard_mode_combo_{nullptr};
  QComboBox * input_mode_combo_{nullptr};

  double aileron_{0.0};
  double elevator_{0.0};
  double throttle_{0.0};
  double rudder_{0.0};
  double aileron_trim_{0.0};
  double elevator_trim_{0.0};
  bool enabled_{true};
  int onboard_mode_{0};  // 0 = manual, 1 = stabilized
  int input_mode_{0};    // 0 = joystick, 1 = auto-autolevel

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<cubs2_msgs::msg::AircraftControl>::SharedPtr joy_publisher_{nullptr};
  rclcpp::Subscription<cubs2_msgs::msg::AircraftControl>::SharedPtr joy_subscriber_{nullptr};
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr mode_publisher_{nullptr};
};

}  // namespace cubs2
