#include "traffic_detection_perception/traffic_detection_perception.h"
#include "dnn_node/dnn_node.h"

#include <algorithm>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "dnn_node/util/image_proc.h"

TrafficDetectionPerceptionNode::
TrafficDetectionPerceptionNode(
    const std::string& node_name,
    const NodeOptions& options)
: DnnNode(node_name,options)
{


    this->declare_parameter(
        "model_path",
        "/userdata/model.bin");


    this->declare_parameter(
        "model_name",
        "model");


    GetParams();


    Init();



    detect_pub_ =
        this->create_publisher<std_msgs::msg::Bool>(
            "traffic_sign_detected",
            10);


    score_pub_ =
        this->create_publisher<std_msgs::msg::Float32>(
            "traffic_sign_score",
            10);



    subscriber_ =
        this->create_subscription<
        hbm_img_msgs::msg::HbmMsg1080P>(

        "hbmem_img",
        rclcpp::SensorDataQoS(),

        std::bind(
            &TrafficDetectionPerceptionNode::
            subscription_callback,

            this,

            std::placeholders::_1));


}

TrafficDetectionPerceptionNode::
~TrafficDetectionPerceptionNode()
{

}

bool TrafficDetectionPerceptionNode::GetParams()
{

    auto client =
        std::make_shared<
        rclcpp::SyncParametersClient>(this);


    auto params =
        client->get_parameters(
        {
            "model_path",
            "model_name"
        });


    return AssignParams(params);

}



bool TrafficDetectionPerceptionNode::AssignParams(
const std::vector<rclcpp::Parameter>& params)
{


    for(auto&p:params)
    {

        if(p.get_name()=="model_path")
            model_path_=p.value_to_string();


        if(p.get_name()=="model_name")
            model_name_=p.value_to_string();

    }


    return true;

}



int TrafficDetectionPerceptionNode::SetNodePara()
{

    dnn_node_para_ptr_->model_file=model_path_;

    dnn_node_para_ptr_->model_name=model_name_;

    dnn_node_para_ptr_->model_task_type=
        model_task_type_;


    dnn_node_para_ptr_->task_num=1;


    return 0;

}

void TrafficDetectionPerceptionNode::subscription_callback(
    const hbm_img_msgs::msg::HbmMsg1080P::SharedPtr msg)
{
    if (!msg || !rclcpp::ok()) {
        return;
    }

    auto model_manage = GetModel();
    if (!model_manage) {
        RCLCPP_ERROR(this->get_logger(), "Invalid model");
        return;
    }

    cv::Mat img_mat(
        msg->height * 3 / 2,
        msg->width,
        CV_8UC1,
        static_cast<void *>(msg->data.data()));

    auto pyramid =
        hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char *>(img_mat.data),
            msg->height,
            msg->width,
            320,
            320);

    if (!pyramid) {
        RCLCPP_ERROR(this->get_logger(), "Get NV12 pyramid failed");
        return;
    }

    std::vector<std::shared_ptr<DNNInput>> inputs;

    auto rois = std::make_shared<std::vector<hbDNNRoi>>();

    hbDNNRoi roi {};
    roi.left = 0;
    roi.top = 0;
    roi.right = 320;
    roi.bottom = 320;

    rois->push_back(roi);

    for (size_t i = 0; i < rois->size(); ++i) {
        for (int32_t j = 0;
             j < model_manage->GetInputCount();
             ++j) {
            inputs.push_back(pyramid);
             }
    }

    // 保持为空，由 Run() 内部创建并触发 PostProcess()
    auto dnn_output = std::shared_ptr<DnnNodeOutput>();

    int ret = Predict(inputs, dnn_output, rois);

    if (ret != 0) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Predict failed, ret: %d",
            ret);
    }
}

int TrafficDetectionPerceptionNode::Predict(
    std::vector<std::shared_ptr<DNNInput>>& inputs,
    const std::shared_ptr<DnnNodeOutput>& output,
    const std::shared_ptr<std::vector<hbDNNRoi>>& rois)
{


    auto ret = Run(
        inputs,
        output,
        rois,
        true);


    if(ret != 0)
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "Run failed");

        return ret;
    }


    return 0;
}

int TrafficParser::Parse(
std::shared_ptr<TrafficResult>& result,
std::shared_ptr<DNNTensor>& tensor)
{


    hbSysFlushMem(
        &(tensor->sysMem[0]),
        HB_SYS_MEM_CACHE_INVALIDATE);



    float* output =
        reinterpret_cast<float*>(
            tensor->sysMem[0].virAddr);



    float max_score=0;



    for(int i=0;i<2100;i++)
    {


        for(int c=0;c<4;c++)
        {

            float score =
                output[
                    (4+c)*2100+i
                ];


            max_score =
                std::max(
                    max_score,
                    score);

        }

    }



    result->score=max_score;


    result->detected =
        max_score > 0.5;



    return 0;
}

int TrafficDetectionPerceptionNode::PostProcess(
const std::shared_ptr<DnnNodeOutput>& outputs)
{


    auto tensor = outputs->output_tensors[0];


    auto result =
        std::make_shared<TrafficResult>();


    result->Reset();



    TrafficParser parser;


    parser.Parse(
        result,
        tensor);



    std_msgs::msg::Bool msg;


    msg.data =
        result->detected;


    detect_pub_->publish(msg);



    std_msgs::msg::Float32 score;


    score.data =
        result->score;


    score_pub_->publish(score);



    return 0;

}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<TrafficDetectionPerceptionNode>(
            "TrafficDetection"));

    rclcpp::shutdown();

    RCLCPP_WARN(
        rclcpp::get_logger("TrafficDetectionPerceptionNode"),
        "Pkg exit.");

    return 0;
}