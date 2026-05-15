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

#include <memory>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/u_int16_multi_array.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "cubs2_msgs/msg/aircraft_control.hpp"
#include <boost/asio.hpp> 
using namespace boost;
using std::placeholders::_1;
using namespace std::chrono_literals;


union servo_data_t{
  char bytes[12];
  uint16_t data[6];
};

class MinimalSubscriber : public rclcpp::Node
{
  public:
    //Public member method
    MinimalSubscriber()
    : Node("minimal_subscriber"), m_io(), m_port(m_io)
    {
      // Declare Joy Controller
      this->declare_parameter("controller_id", rclcpp::PARAMETER_STRING);
      rclcpp::Parameter controller_param = this->get_parameter("controller_id");

      m_controller_id_param_str = controller_param.as_string();

      // Declare Joy msg type subscription for manual mode
      m_subscription = this->create_subscription<sensor_msgs::msg::Joy>(
      "joy_throttle", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));

      // Declare AircraftControl msg type subscription for auto mode
      a_subscription = this->create_subscription<cubs2_msgs::msg::AircraftControl>(
      "control_auto_throttle", 10, std::bind(&MinimalSubscriber::auto_callback, this, _1));

      m_pub_status = this->create_publisher<std_msgs::msg::String>("status", 10);
      joy_pub_status = this->create_publisher<std_msgs::msg::UInt16MultiArray>("joy_serial_status", 10);
      // Publishes current outerloop mode ("manual"/"auto") so RViz JoyPanel can reflect the switch position
      outerloop_mode_pub_ = this->create_publisher<std_msgs::msg::String>("/toggle_outerloop_mode", 10);


      auto get_status =
      [this]() -> void
      {
        memset(m_out_buf, 0, 2048);
        size_t n_read = m_port.read_some(asio::buffer(m_out_buf,2048));
        m_out_buf[n_read] = '\0';
        RCLCPP_INFO(this->get_logger(), "%s", m_out_buf);
      };
      //m_timer  = this->create_wall_timer(100ms, get_status);

      try {
        m_port.open("/dev/ttyACM0");
        m_port.set_option(asio::serial_port_base::baud_rate(57600));
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open serial port /dev/ttyACM0: %s", e.what());
        throw;
      }
      for (int i = 0; i<5; i++){
        m_servo_data.data[i] = 1000;
      }  
    }
    ~MinimalSubscriber(){
        m_port.close();
    }
    //public member attributes

  private:
    // Private member methodsjoy
    void topic_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
      RCLCPP_DEBUG(this->get_logger(), "controller id: %s", m_controller_id_param_str.c_str());
      if (m_controller_id_param_str == "taranis"){

        m_servo_data.data[0] = std::clamp(-500.0 * msg->axes[0] + 1500.0, 1000.0, 2000.0); // joy "+1" is zero throttle , joy "-1" is full throttle
        m_servo_data.data[1] = std::clamp(500.0 * msg->axes[1] + 1500.0, 1000.0, 2000.0); // joy "+1" is roll left, joy "-1" is roll right
        m_servo_data.data[2] = std::clamp(-500.0 * msg->axes[2] + 1500.0, 1000.0, 2000.0); // joy "+1" is elev up (pitch up), joy "-1" elev down (pitch down)
        m_servo_data.data[3] = std::clamp(500.0 * msg->axes[3] + 1500.0, 1000.0, 2000.0); // joy "+1" is rudder yaw left, joy "-1 is yaw right"
        m_servo_data.data[4] = std::clamp(1000.0 * msg->axes[4] + 2000.0, 1000.0, 2000.0);
        
        // Taranis switch-type mode handling
        if (msg->axes[5] > 0){
          m_servo_data.data[0] = a_servo_data.data[0];
          m_servo_data.data[1] = a_servo_data.data[1];
          m_servo_data.data[2] = a_servo_data.data[2];
          m_servo_data.data[3] = a_servo_data.data[3];
          m_servo_data.data[4] = a_servo_data.data[4];
        }
      }
      // else, this will work with f310 logitech
      else {
        // button-type mode handling
        if (msg->buttons[0] == 1 && !toggle_mode_switch_){
          is_auto_mode_ = !is_auto_mode_;
          toggle_mode_switch_ = true;
        } else if (msg->buttons[0] == 0){
          toggle_mode_switch_ = false; // reset button switch when released
        }

        m_servo_data.data[0] = std::clamp(1000.0 * msg->axes[1] + 1000, 1000.0, 2000.0);
        m_servo_data.data[1] = std::clamp(500.0 * msg->axes[3] + 1500, 1000.0, 2000.0);
        m_servo_data.data[2] = std::clamp(500.0 * msg->axes[4] + 1500, 1000.0, 2000.0);
        m_servo_data.data[3] = std::clamp(500.0 * msg->axes[0] + 1500, 1000.0, 2000.0);
        m_servo_data.data[4] = std::clamp(1000.0 * msg->axes[2] + 2000, 1000.0, 2000.0); // stabilizer mode
        
        // Logitech f310 mode switch using "A" button
        if (is_auto_mode_){
          m_servo_data.data[0] = a_servo_data.data[0];
          m_servo_data.data[1] = a_servo_data.data[1];
          m_servo_data.data[2] = a_servo_data.data[2];
          m_servo_data.data[3] = a_servo_data.data[3];
          m_servo_data.data[4] = a_servo_data.data[4];
        }
      }

      // Determine current switch state and publish mode change to RViz when it changes
      bool current_is_auto = (m_controller_id_param_str == "taranis")
                              ? (msg->axes[5] > 0)
                              : is_auto_mode_;
      if (current_is_auto != prev_is_auto_) {
        prev_is_auto_ = current_is_auto;
        std_msgs::msg::String mode_msg;
        mode_msg.data = current_is_auto ? "auto" : "manual";
        outerloop_mode_pub_->publish(mode_msg);
        RCLCPP_INFO(this->get_logger(), "Outerloop mode: %s", mode_msg.data.c_str());
      }

      uint16_t cksum = 0;
      for (int i=0;i<5;i++) {
        cksum += m_servo_data.data[i];
      }
      char packet[14];
      uint16_t header = 65535;
      memcpy(packet, &header, 2);
      memcpy(&packet[2], m_servo_data.bytes, 10);
      memcpy(&packet[12], &cksum, 2);
      m_port.write_some(asio::buffer(packet, 14));

      std_msgs::msg::String status;
      char buf[255];
      snprintf(buf, 255,
        "Sent to serial [AETR+Mode]: [%u, %u, %u, %u, %u]", // Messages Sent to Serial is in AETR+Mode format
          m_servo_data.data[1],
          m_servo_data.data[2],
          m_servo_data.data[0],
          m_servo_data.data[3],
          m_servo_data.data[4]);
      status.data = std::string(buf);
      m_pub_status->publish(status);

      std_msgs::msg::UInt16MultiArray joy_arr; //joy array message to publish to Arduino in AETR mode
      joy_arr.data = {
        m_servo_data.data[1],
          m_servo_data.data[2],
          m_servo_data.data[0],
          m_servo_data.data[3],
          m_servo_data.data[4]};
      joy_pub_status->publish(joy_arr);

    }
     void auto_callback(const cubs2_msgs::msg::AircraftControl::SharedPtr msg) // Designs to take AircraftControl and pack into Spektrum's PPM "TAER" format
    {
      a_servo_data.data[0] = std::clamp(1000.0 * msg->throttle + 1000.0, 1000.0, 2000.0); // Throttle [0,1]
      a_servo_data.data[1] = std::clamp(500.0 * msg->aileron + 1500.0, 1000.0, 2000.0);   // Aileron [-1,1]
      a_servo_data.data[2] = std::clamp(-500.0 * msg->elevator + 1500.0, 1000.0, 2000.0); // Elevator [-1,1]
      a_servo_data.data[3] = std::clamp(500.0 * msg->rudder + 1500.0, 1000.0, 2000.0);    // Rudder [-1,1]
      a_servo_data.data[4] = 2000.0; // force stabilize mode
    }

    // Member attributes
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr m_subscription;
    rclcpp::Subscription<cubs2_msgs::msg::AircraftControl>::SharedPtr a_subscription;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_pub_status;
    rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr joy_pub_status;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr outerloop_mode_pub_;

    asio::io_service m_io;
    asio::serial_port m_port;
    servo_data_t m_servo_data;
    servo_data_t a_servo_data;
    
    // Joy button mode handling for Logitech
    bool is_auto_mode_{false};
    bool toggle_mode_switch_{false};
    bool prev_is_auto_{false};

    u_int8_t m_out_buf[2048];
    rclcpp::TimerBase::SharedPtr m_timer;
    std::string m_controller_id_param_str;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MinimalSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}