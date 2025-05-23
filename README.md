### "社交图谱链接预测" 背景

社会网络是由社会个体成员之间因为互动而形成的相对稳定的社会结构，成员之间的互动和联系进一步影响人们的社会行为，电子商务平台大范围的普及和使用，不仅满足人们丰富多样的消费需求，也承载着社会成员基于商品消费产生的互动链接，形成基于电商平台的在线社交网络，电商场景社交知识图谱的构建有助于深入理解在线社交网络的结构特性与演化机理，为用户社交属性识别和互动规律发现提供有效方式。电商平台活动和场景形式丰富多样，用户表现出不同的社交行为偏好，且伴随活动场景、互动对象、互动方式、互动时间的不同而不断发生变化，动态性高，不确定性强，这些都给社交知识图谱的构建和应用带来巨大挑战。

任务描述
----

商品分享是用户在电商平台中主要的社交互动形式，商品分享行为包含了用户基于商品的社交互动偏好信息，用户的商品偏好和社交关系随时间变化，其基于商品的社交互动属性也随时间改变。本任务关注社交图谱动态链接预测问题，用四元组 （u， i， v， t） 表示用户在 t （time） 时刻的商品分享互动行为，其中 i （item） 标识特定商品，u （user） 表示发起商品链接分享的邀请用户，v （voter） 表示接收并点击该商品链接的回流用户。因此在本任务中链接预测指的是，在已知邀请用户 u，商品 i 和时间 t 的情况下，预测对应回流用户 v。

任务目标
----

针对社交图谱动态链接预测，参赛队伍需要根据已有动态社交图谱四元组数据，对于给定 u，i，t，对 v 进行预测。

数据集介绍
-----

### 初赛训练集 & 测试集

在初赛阶段，我们会发布用户动态商品分享数据作为训练集。训练集由三部分构成，可以在天池平台下载获取。

*   item_share_train_info.json：用户动态商品分享数据，每行是一个 json 串，具体字段信息如下：

<table><thead><tr><th><strong _msttexthash="4995445" _msthash="268">字段</strong></th><th><strong _msttexthash="12583701" _msthash="269">字段说明</strong></th></tr></thead><tbody><tr><td _msttexthash="96109" _msthash="270">user_id</td><td _msttexthash="13891397" _msthash="271">邀请用户 ID</td></tr><tr><td _msttexthash="94471" _msthash="272">item_id</td><td _msttexthash="9401041" _msthash="273">分享商品 ID</td></tr><tr><td _msttexthash="116051" _msthash="274">voter_id</td><td _msttexthash="17299672" _msthash="275">选民的用户 ID</td></tr><tr><td _msttexthash="9308897" _msthash="276">时间戳</td><td _msttexthash="29834571" _msthash="277">分享行为发生时间</td></tr></tbody></table>

*   user_info.json：用户信息数据，每行是一个 jsonl 串，具体字段信息如下：

<table><thead><tr><th><strong _msttexthash="4995445" _msthash="279">字段</strong></th><th><strong _msttexthash="12583701" _msthash="280">字段说明</strong></th></tr></thead><tbody><tr><td _msttexthash="96109" _msthash="281">user_id</td><td _msttexthash="81314493" _msthash="282">用户的 ID，包含数据集中所有的 inviter_id 和 voter_id</td></tr><tr><td _msttexthash="181688" _msthash="283">user_gender</td><td _msttexthash="175346587" _msthash="284">用户性别，0 表示女性用户，1 表示男性用户，未知为 - 1</td></tr><tr><td _msttexthash="113750" _msthash="285">user_age</td><td _msttexthash="276977467" _msthash="286">用户所在年龄段，数值范围为 1～8，数值越大表示年龄越大，未知为 - 1</td></tr><tr><td _msttexthash="160381" _msthash="287">user_level</td><td _msttexthash="319972133" _msthash="288">用户的平台积分水平，数值范围为 1~10，数值越大表示用户的平台积分水平越高</td></tr></tbody></table>

*   item_info.json：商品信息数据，每行是一个 json 串，具体字段信息如下：

<table><thead><tr><th><strong _msttexthash="4995445" _msthash="290">字段</strong></th><th><strong _msttexthash="12583701" _msthash="291">字段说明</strong></th></tr></thead><tbody><tr><td _msttexthash="94471" _msthash="292">item_id</td><td _msttexthash="10843846" _msthash="293">商品编号</td></tr><tr><td _msttexthash="92664" _msthash="294">cate_id</td><td _msttexthash="19127966" _msthash="295">商品叶子类目 ID</td></tr><tr><td _msttexthash="241930" _msthash="296">cate_level1_id</td><td _msttexthash="20125534" _msthash="297">商品一级类目 ID</td></tr><tr><td _msttexthash="111488" _msthash="298">brand_id</td><td _msttexthash="22954620" _msthash="299">商品所属的品牌 ID</td></tr><tr><td _msttexthash="95693" _msthash="300">shop_id</td><td _msttexthash="24847810" _msthash="301">商品所属的店铺 ID</td></tr></tbody></table>

在初赛阶段，我们会发布测试集 item_share_ preliminary_test_info.json，每行是一个 json 串，具体字段信息如下：

<table><thead><tr><th><strong _msttexthash="4995445" _msthash="303">字段</strong></th><th><strong _msttexthash="12583701" _msthash="304">字段说明</strong></th></tr></thead><tbody><tr><td _msttexthash="96109" _msthash="305">user_id</td><td _msttexthash="13891397" _msthash="306">邀请用户 ID</td></tr><tr><td _msttexthash="94471" _msthash="307">item_id</td><td _msttexthash="9401041" _msthash="308">分享商品 ID</td></tr><tr><td _msttexthash="9308897" _msthash="309">时间戳</td><td _msttexthash="29834571" _msthash="310">分享行为发生时间</td></tr></tbody></table>

### 复赛测试集

在复赛阶段我们发布复赛测试集 item_share_final_test_info.json，复赛结果提交方式和初赛保持一致，具体参见任务提交说明章节。

任务提交说明
------

参赛队伍可在初赛和复赛阶段下载对应的测试集数据。每一行包括 triple_id， inviter_id， item_id， timestamp。

测试集示例如下： {“triple_id”：“0”， “inviter_id”：“0”， “item_id”：“12423”，

“timestamp”：“2023-01-05 10：11：  
12”

}

本次比赛的目标对给定的邀请者 inviter_id，物品 item_id 和时间信息时间戳，预测此次交互行为的回流者 voter_id。参赛队伍需要从用户信息数据 （见 user_info.json） 中选择用户，给出一个长度为 5 的候选回流者 （voter_id） 列表，其中预测概率大的 voter 排在前面，把对应的 user_id 填入 candidate_voter_list，最后以 json 文件的形式提交。初赛和复赛的提交文件形式相同。

提交文件示例如下：

```
{
	"triple_id":"0",
	"candidate_voter_list": ["1", "2", "3", "4", "5"]
}


```

评价指标
----

本任务采用 MRR（Mean Reciprocal Rank，平均倒数排名）指标来评估模型的预测效果。

对于一个查询：（inviter， item， ？， timestamp ），我们需要根据概率找出可能的候选选民列表，为 candidate_voter_list，概率高的排在前面。若真实的回流者排在第 n 位，则该次 query 的分数为 1/n。由于候选 voter 列表长度为 5，所以此处 n 最大为 5，如果列表中不包含真实的回流者，则此次查询的得分为 0。最终对多个查询的分数求平均，计算公式如下：

M R R=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{{rank}_i}

其中，为测试 query 集合，![](https://intranetproxy.alipay.com/skylark/lark/__latex/4ef7132d0df72d9e3db76f6391960a3d.svg)![](https://intranetproxy.alipay.com/skylark/lark/__latex/e0855a1809001422510099ea59ad348c.svg)为其中 query 的个数，  
\frac{1}{\operatorname{rank}_i}= \begin{cases}\frac{1}{i} & \text { （真实回流者排在第 } i \text { 位） } \\ 0 & \text {（ 候选选民列表中没有真实的回流者）}\end{cases}

### 基线模型

<table><thead><tr><th></th><th _msttexthash="41028" _msthash="324">MRR@5</th><th _msttexthash="52182" _msthash="325">HITS@5</th></tr></thead><tbody><tr><td><a href="https://github.com/meteor-gif/BDSC_Task_2" target="_blank" _msttexthash="5423990" _msthash="326">基线</a></td><td _msttexthash="46085" _msthash="327">0.03437</td><td _msttexthash="47060" _msthash="328">0.09258</td></tr></tbody></table>

