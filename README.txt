包含两个功能:a.对一张图片构建SIFT特征点并画出特征点 b.对两张图片构建SIFT特征点并匹配
存在的问题：
1.描述子没有插值因此没有确定主方向的精确位置，但我觉得插值影响不大，因为尺度是离散且较小的(scale=36)
2.上采样用的pyrup，但是不是严格二倍关系，精确定位时存在一定误差
3.精确定位时没有改变尺度i，因为调试过程中尺度i总是会收敛到边界值
4.描述子没有在金字塔上构建，而是直接在原图中构建了，对尺度变化大的匹配效果不好
5.直接对灰度图构建金字塔，没有构建RGB金字塔
6.没有采用单应变换进行ransac匹配筛选，而是使用描述子进行one-to-one的匹配，误差采用的两个特征点在sift特征空间的欧氏距离
7.参数可能没有调正确，包括检测极值点时的阈值，高斯模糊的方差，匹配loss的阈值，方向尺度scale的选择
8.没有进行边缘剔除，但我觉得如果进行了边缘剔除效果会更不好
