from .detector3d_template import Detector3DTemplate

# 功能：基于Detector3DTemplate构建PointPillar算法结构
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        """
        Args:
            model_cfg:   yaml配置文件的MODEL部分
            num_class:   类别数目（kitti数据集一般用3个类别：'Car', 'Pedestrian', 'Cyclist'）
            dataset:     训练数据集
        """
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)   # 初始化基类

        # 网络的各处理模块已经存储在self中(vfe / map_to_bev / backbone_2d ...)
        self.module_list = self.build_networks()    # 真正构建模型的处理函数，Detector3DTemplate的子函数

    def forward(self, batch_dict):
        # 各模块分别进行特征处理，更新batch_dict，然后将预测信息与gt信息保存在forward_ret_dict字典中来进行后续的损失计算
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)     # 在dense_head中训练与测试的batch_dict存在区别

        # 训练过程进行损失计算
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()    # 损失计算

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        # 测试过程进行后处理返回预测结果
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)  # pred_dicts:{list:16}、 recall_dicts:{dict:7}
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        """
        在进行各模块处理后，在forward函数中进行损失计算
        """
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()  # AnchorHeadTemplate.get_loss中执行
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict   # **保留key和values
        }

        loss = loss_rpn

        # 增加cornor loss损失
        if self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NAME == 'AxisAlignedTargetAssignerCornorLoss':
            reg_cornor_loss = self.dense_head.forward_ret_dict['reg_cornor_loss']   # (16, 321408)
            reg_weights = self.dense_head.forward_ret_dict['reg_weights']    # (16, 321408)
            cornor_loss = reg_cornor_loss.sum() / reg_weights.sum()     # 角点损失累加以及用角点数进行归一化
            loss += cornor_loss / reg_weights.shape[0]     # 对batch的cornor loss进行求和
        return loss, tb_dict, disp_dict
