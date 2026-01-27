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
#include "cubs2_rviz/joy_panel.hpp"
#include <QComboBox>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <pluginlib/class_list_macros.hpp>
#include "cubs2_msgs/msg/aircraft_control.hpp"

namespace cubs2
{

// ============================================================================
// Joy Panel - Virtual Joystick Control
// Publishes AircraftControl messages to /control topic
// ============================================================================

// ============================================================================
// JoystickWidget Implementation
// ============================================================================

JoystickWidget::JoystickWidget(QWidget * parent)
: QWidget(parent)
{
  setMinimumSize(150, 150);
  setMouseTracking(false);

  // Spring-back timer (50ms = 20 Hz smooth return to center)
  spring_timer_ = new QTimer(this);
  spring_timer_->setInterval(50);
  connect(spring_timer_, &QTimer::timeout, this, &JoystickWidget::springBackStep);
}

void JoystickWidget::paintEvent(QPaintEvent * event)
{
  (void)event;
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);

  int w = width();
  int h = height();
  int cx = w / 2;
  int cy = h / 2;
  int radius = qMin(w, h) / 2 - 10;

  // Draw background circle
  painter.setBrush(QColor(50, 50, 50));
  painter.setPen(QPen(QColor(100, 100, 100), 2));
  painter.drawEllipse(cx - radius, cy - radius, 2 * radius, 2 * radius);

  // Draw trim center point (if trim is non-zero)
  if (aileron_trim_ != 0.0 || elevator_trim_ != 0.0) {
    int trim_x = cx + static_cast<int>(aileron_trim_ * radius);
    int trim_y = cy + static_cast<int>(elevator_trim_ * radius);  // Match elevator sign convention
    painter.setBrush(QColor(100, 150, 255, 128));
    painter.setPen(QPen(QColor(80, 120, 200), 1));
    painter.drawEllipse(trim_x - 4, trim_y - 4, 8, 8);
  }

  // Draw crosshairs
  painter.setPen(QPen(QColor(150, 150, 150), 1, Qt::DashLine));
  painter.drawLine(cx - radius, cy, cx + radius, cy);  // Horizontal
  painter.drawLine(cx, cy - radius, cx, cy + radius);  // Vertical

  // Draw joystick position
  int stick_x = cx + static_cast<int>(aileron_ * radius);
  int stick_y =
    cy + static_cast<int>(elevator_ * radius);    // Positive elev = down on screen (pull back)

  painter.setBrush(QColor(255, 100, 100));
  painter.setPen(QPen(QColor(200, 50, 50), 2));
  painter.drawEllipse(stick_x - 8, stick_y - 8, 16, 16);

  // Draw labels
  painter.setPen(QColor(200, 200, 200));
  painter.drawText(10, 20, "Aileron/Elevator");
  painter.drawText(10, h - 10,
                   QString("A: %1  E: %2").arg(aileron_, 0, 'f', 2).arg(elevator_, 0, 'f', 2));
}

void JoystickWidget::mousePressEvent(QMouseEvent * event)
{
  if (event->button() == Qt::LeftButton) {
    dragging_ = true;
    updatePosition(event->pos());
  }
}

void JoystickWidget::mouseMoveEvent(QMouseEvent * event)
{
  if (dragging_) {
    updatePosition(event->pos());
  }
}

void JoystickWidget::mouseReleaseEvent(QMouseEvent * event)
{
  if (event->button() == Qt::LeftButton) {
    dragging_ = false;
    // Start spring-back to center
    startSpringBack();
  }
}

void JoystickWidget::startSpringBack()
{
  spring_timer_->start();
}

void JoystickWidget::springBackStep()
{
  // Exponential decay toward zero with rate constant
  constexpr double decay_rate = 0.2;  // Higher = faster return

  aileron_ *= (1.0 - decay_rate);
  elevator_ *= (1.0 - decay_rate);

  // Stop when very close to zero
  if (std::abs(aileron_) < 0.01 && std::abs(elevator_) < 0.01) {
    aileron_ = 0.0;
    elevator_ = 0.0;
    spring_timer_->stop();
  }

  update();
  Q_EMIT joystickMoved(aileron_, elevator_);
}

void JoystickWidget::updatePosition(const QPoint & pos)
{
  int w = width();
  int h = height();
  int cx = w / 2;
  int cy = h / 2;
  int radius = qMin(w, h) / 2 - 10;

  // Convert mouse position to -1..1 range
  double dx = (pos.x() - cx) / static_cast<double>(radius);
  double dy =
    (pos.y() - cy) / static_cast<double>(radius);    // No inversion: down = positive (pull back)

  // Clamp to unit circle
  double mag = std::sqrt(dx * dx + dy * dy);
  if (mag > 1.0) {
    dx /= mag;
    dy /= mag;
  }

  aileron_ = dx;
  elevator_ = dy;

  // Stop spring-back timer if dragging
  if (spring_timer_->isActive()) {
    spring_timer_->stop();
  }

  update();
  Q_EMIT joystickMoved(aileron_, elevator_);
}

// ============================================================================
// JoyPanel Implementation
// ============================================================================

JoyPanel::JoyPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout;

  // Title and enable checkbox
  auto * header_layout = new QHBoxLayout;
  auto * title_label = new QLabel("<b>Virtual Joystick</b>");
  header_layout->addWidget(title_label);
  header_layout->addStretch();
  enable_checkbox_ = new QCheckBox("Enabled");
  enable_checkbox_->setChecked(false);
  header_layout->addWidget(enable_checkbox_);
  layout->addLayout(header_layout);
  connect(enable_checkbox_, &QCheckBox::stateChanged, this, &JoyPanel::onEnabledChanged);

  // Onboard Mode selector
  auto * mode_layout = new QHBoxLayout;
  mode_layout->addWidget(new QLabel("Onboard Mode:"));
  onboard_mode_combo_ = new QComboBox();
  onboard_mode_combo_->addItem("Manual");
  onboard_mode_combo_->addItem("Stabilized");
  onboard_mode_combo_->setCurrentIndex(0);
  mode_layout->addWidget(onboard_mode_combo_);
  
  // Input Mode selector (Joystick / Auto-autolevel)
  mode_layout->addWidget(new QLabel("Input Mode:"));
  input_mode_combo_ = new QComboBox();
  input_mode_combo_->addItem("Joystick");
  input_mode_combo_->addItem("Auto-autolevel");
  input_mode_combo_->setCurrentIndex(0);
  mode_layout->addWidget(input_mode_combo_);
  mode_layout->addStretch();
  layout->addLayout(mode_layout);
  connect(onboard_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &JoyPanel::onOnboardModeChanged);
  connect(input_mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &JoyPanel::onInputModeChanged);

  // 2D Joystick for aileron/elevator
  joystick_ = new JoystickWidget(this);
  layout->addWidget(joystick_);
  connect(joystick_, &JoystickWidget::joystickMoved, this, &JoyPanel::onJoystickMoved);

  // Throttle slider
  auto * throttle_layout = new QHBoxLayout;
  throttle_layout->addWidget(new QLabel("Throttle:"));
  throttle_slider_ = new QSlider(Qt::Horizontal);
  throttle_slider_->setRange(0, 100);
  throttle_slider_->setValue(0);
  throttle_layout->addWidget(throttle_slider_);
  throttle_label_ = new QLabel("0.00");
  throttle_label_->setMinimumWidth(40);
  throttle_layout->addWidget(throttle_label_);
  layout->addLayout(throttle_layout);
  connect(throttle_slider_, &QSlider::valueChanged, this, &JoyPanel::onThrottleChanged);

  // Rudder slider
  auto * rudder_layout = new QHBoxLayout;
  rudder_layout->addWidget(new QLabel("Rudder:"));
  rudder_slider_ = new QSlider(Qt::Horizontal);
  rudder_slider_->setRange(-100, 100);
  rudder_slider_->setValue(0);
  rudder_layout->addWidget(rudder_slider_);
  rudder_label_ = new QLabel("0.00");
  rudder_label_->setMinimumWidth(40);
  rudder_layout->addWidget(rudder_label_);
  layout->addLayout(rudder_layout);
  connect(rudder_slider_, &QSlider::valueChanged, this, &JoyPanel::onRudderChanged);

  // Aileron trim slider (bottom of joystick area)
  auto * aileron_trim_layout = new QHBoxLayout;
  aileron_trim_layout->addWidget(new QLabel("Ail Trim:"));
  aileron_trim_slider_ = new QSlider(Qt::Horizontal);
  aileron_trim_slider_->setRange(-50, 50);  // ±0.5 trim range
  aileron_trim_slider_->setValue(0);
  aileron_trim_layout->addWidget(aileron_trim_slider_);
  aileron_trim_label_ = new QLabel("0.00");
  aileron_trim_label_->setMinimumWidth(40);
  aileron_trim_layout->addWidget(aileron_trim_label_);
  layout->addLayout(aileron_trim_layout);
  connect(aileron_trim_slider_, &QSlider::valueChanged, this, &JoyPanel::onAileronTrimChanged);

  // Elevator trim slider (right of joystick area)
  auto * elevator_trim_layout = new QHBoxLayout;
  elevator_trim_layout->addWidget(new QLabel("Elev Trim:"));
  elevator_trim_slider_ = new QSlider(Qt::Horizontal);
  elevator_trim_slider_->setRange(-50, 50);  // ±0.5 trim range
  elevator_trim_slider_->setValue(0);
  elevator_trim_layout->addWidget(elevator_trim_slider_);
  elevator_trim_label_ = new QLabel("0.00");
  elevator_trim_label_->setMinimumWidth(40);
  elevator_trim_layout->addWidget(elevator_trim_label_);
  layout->addLayout(elevator_trim_layout);
  connect(elevator_trim_slider_, &QSlider::valueChanged, this, &JoyPanel::onElevatorTrimChanged);

  // Reset button
  reset_button_ = new QPushButton("Reset Controls");
  layout->addWidget(reset_button_);
  connect(reset_button_, &QPushButton::clicked, this, &JoyPanel::onResetClicked);

  layout->addStretch();
  setLayout(layout);

  // Create ROS2 node and publishers
  node_ = std::make_shared<rclcpp::Node>("joy_panel");
  joy_publisher_ = node_->create_publisher<cubs2_msgs::msg::AircraftControl>("/control", 10);
  mode_publisher_ = node_->create_publisher<std_msgs::msg::Float32>("/control_mode", 10);

  // Create subscribers to monitor external control (e.g., from gamepad)
  joy_subscriber_ = node_->create_subscription<cubs2_msgs::msg::AircraftControl>(
      "/control", 10,
    [this](const cubs2_msgs::msg::AircraftControl::SharedPtr msg) {controlCallback(msg);});

  // Create timer for publishing control inputs at 20 Hz
  control_timer_ = new QTimer(this);
  control_timer_->setInterval(50);  // 20 Hz
  connect(control_timer_, &QTimer::timeout, this, &JoyPanel::publishControlInputs);
  control_timer_->start();

  // Create timer for spinning ROS2 node (process callbacks in Qt thread)
  ros_spin_timer_ = new QTimer(this);
  ros_spin_timer_->setInterval(10);  // 100 Hz for responsive callbacks
  connect(ros_spin_timer_, &QTimer::timeout, this, [this]() {rclcpp::spin_some(node_);});
  ros_spin_timer_->start();

  // Set initial state after all widgets are created
  onEnabledChanged(enable_checkbox_->isChecked() ? Qt::Checked : Qt::Unchecked);
}

JoyPanel::~JoyPanel() = default;

void JoyPanel::onJoystickMoved(double aileron, double elevator)
{
  aileron_ = aileron;
  elevator_ = elevator;
  // Publishing happens in the timer callback
}

void JoyPanel::onThrottleChanged(int value)
{
  throttle_ = value / 100.0;
  throttle_label_->setText(QString::number(throttle_, 'f', 2));
}

void JoyPanel::onRudderChanged(int value)
{
  rudder_ = value / 100.0;
  rudder_label_->setText(QString::number(rudder_, 'f', 2));
}

void JoyPanel::onAileronTrimChanged(int value)
{
  aileron_trim_ = value / 100.0;  // ±0.5 range
  aileron_trim_label_->setText(QString::number(aileron_trim_, 'f', 2));
  joystick_->setTrim(aileron_trim_, elevator_trim_);
}

void JoyPanel::onElevatorTrimChanged(int value)
{
  elevator_trim_ = value / 100.0;  // ±0.5 range
  elevator_trim_label_->setText(QString::number(elevator_trim_, 'f', 2));
  joystick_->setTrim(aileron_trim_, elevator_trim_);
}

void JoyPanel::publishControlInputs()
{
  // Only publish if enabled
  if (!enabled_) {
    return;
  }

  // Publish AircraftControl message
  cubs2_msgs::msg::AircraftControl control_msg;
  control_msg.header.stamp = node_->now();
  control_msg.aileron = static_cast<float>(aileron_ + aileron_trim_);
  control_msg.elevator = static_cast<float>(elevator_ + elevator_trim_);
  control_msg.throttle = static_cast<float>(throttle_);
  control_msg.rudder = static_cast<float>(rudder_);
  control_msg.mode = static_cast<uint8_t>(onboard_mode_);
  joy_publisher_->publish(control_msg);

  // Also publish mode to /control_mode topic for backward compatibility
  std_msgs::msg::Float32 mode_msg;
  mode_msg.data = static_cast<float>(onboard_mode_);
  mode_publisher_->publish(mode_msg);
}

void JoyPanel::onResetClicked()
{
  joystick_->reset();
  throttle_slider_->setValue(0);
  rudder_slider_->setValue(0);
  aileron_trim_slider_->setValue(0);
  elevator_trim_slider_->setValue(0);
  aileron_ = 0.0;
  elevator_ = 0.0;
  throttle_ = 0.0;
  rudder_ = 0.0;
  aileron_trim_ = 0.0;
  elevator_trim_ = 0.0;
}

void JoyPanel::onEnabledChanged(int state)
{
  enabled_ = (state == Qt::Checked);
  // Enable/disable controls for user input, but keep them visually active
  // so they can display external control values
  throttle_slider_->setEnabled(enabled_);
  rudder_slider_->setEnabled(enabled_);
  aileron_trim_slider_->setEnabled(enabled_);
  elevator_trim_slider_->setEnabled(enabled_);
  reset_button_->setEnabled(enabled_);
  onboard_mode_combo_->setEnabled(enabled_);
  input_mode_combo_->setEnabled(enabled_);

  // Don't disable the joystick widget - it needs to display external control
  // Just make it non-interactive when disabled
  joystick_->setAttribute(Qt::WA_TransparentForMouseEvents, !enabled_);
}

void JoyPanel::onOnboardModeChanged(int index)
{
  onboard_mode_ = index;  // 0 = manual, 1 = stabilized
  
  // Publish mode change immediately when user toggles the combobox
  if (node_) {
    // Publish full AircraftControl message with updated mode
    cubs2_msgs::msg::AircraftControl control_msg;
    control_msg.header.stamp = node_->now();
    control_msg.aileron = static_cast<float>(aileron_ + aileron_trim_);
    control_msg.elevator = static_cast<float>(elevator_ + elevator_trim_);
    control_msg.throttle = static_cast<float>(throttle_);
    control_msg.rudder = static_cast<float>(rudder_);
    control_msg.mode = static_cast<uint8_t>(onboard_mode_);
    joy_publisher_->publish(control_msg);
    
    // Also publish mode to /control_mode topic for backward compatibility
    std_msgs::msg::Float32 mode_msg;
    mode_msg.data = static_cast<float>(onboard_mode_);
    mode_publisher_->publish(mode_msg);
  }
}

void JoyPanel::onInputModeChanged(int index)
{
  input_mode_ = index;  // 0 = joystick, 1 = auto-autolevel
  // Input mode selection would be handled by controllers
  // (gamepad passes through vs autolevel commands)
}

void JoyPanel::controlCallback(const cubs2_msgs::msg::AircraftControl::SharedPtr msg)
{
  // Update mode combobox if changed from external source (gamepad, etc.)
  if (static_cast<int>(msg->mode) != onboard_mode_) {
    onboard_mode_ = static_cast<int>(msg->mode);
    onboard_mode_combo_->blockSignals(true);
    onboard_mode_combo_->setCurrentIndex(onboard_mode_);
    onboard_mode_combo_->blockSignals(false);
  }
  
  // Always update display to show current control inputs
  updateDisplayFromExternal(msg->aileron, msg->elevator, msg->throttle, msg->rudder);
}

void JoyPanel::updateDisplayFromExternal(
  double aileron, double elevator, double throttle,
  double rudder)
{
  // Update internal state (for display purposes)
  aileron_ = aileron;
  elevator_ = elevator;
  throttle_ = throttle;
  rudder_ = rudder;

  // Block signals to prevent feedback loop
  throttle_slider_->blockSignals(true);
  rudder_slider_->blockSignals(true);

  // Update UI elements to show external control values
  throttle_slider_->setValue(static_cast<int>(throttle * 100));
  rudder_slider_->setValue(static_cast<int>(rudder * 100));
  throttle_label_->setText(QString::number(throttle, 'f', 2));
  rudder_label_->setText(QString::number(rudder, 'f', 2));

  throttle_slider_->blockSignals(false);
  rudder_slider_->blockSignals(false);

  // Update joystick widget position to show external control
  // The joystick display shows the stick position without trim
  double ail_stick = aileron - aileron_trim_;
  double elev_stick = elevator - elevator_trim_;
  joystick_->setPosition(ail_stick, elev_stick);
}

}  // namespace cubs2

PLUGINLIB_EXPORT_CLASS(cubs2::JoyPanel, rviz_common::Panel)
