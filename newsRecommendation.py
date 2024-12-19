import collections
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 配置类，用于管理数据路径相关的配置信息
class Config:
    def __init__(self):
        # 原始数据存储路径，用于读取训练集和测试集的点击数据等文件
        self.data_path = 'D:/SchoolStudy/深度学习/课程设计数据/'
        # 临时结果保存路径，例如生成的提交文件等会保存在此路径下
        self.save_path = 'D:/SchoolStudy/深度学习/课程设计数据/temp_results/'


# 获取点击数据，将训练集和测试集的点击数据合并，并去除重复的用户、文章和点击时间组合
def get_all_click_df():
    config = Config()
    try:
        # 分别读取训练集点击数据文件
        trn_click = pd.read_csv(config.data_path + 'train_click_log.csv')
        # 读取测试集点击数据文件
        tst_click = pd.read_csv(config.data_path + 'testA_click_log.csv')
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"数据解析错误: {e}")
        raise

    # 把训练集和测试集的数据合并到一起
    all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    # 去除重复的用户、文章和点击时间组合，保证数据的唯一性
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

    # 使用sample方法随机抽取250000条数据，如果数据总量小于250000，则全部返回，这里可适当调整抽样数量
    num_samples = min(len(all_click), 250000)
    return all_click.sample(n=num_samples, random_state=42)  # 设置random_state保证每次抽样结果可复现


# !!!!!!!!!!!!!!!!!!!!!!!!!
# 自定义数据集类，用于处理点击数据，使其能被深度学习模型使用
class ClickDataset(Dataset):
    def __init__(self, all_click_df):
        # 获取用户 - 文章 - 时间序列的字典，用于后续处理
        self.user_item_time_dict = self.get_user_item_time(all_click_df)
        # 获取所有用户的id列表
        self.user_ids = list(self.user_item_time_dict.keys())
        # 获取所有文章的id列表，通过去重得到
        self.item_ids = list(set([item for sublist in self.user_item_time_dict.values() for item, _ in sublist]))
        # 构建用户id到索引的映射字典，方便后续将用户id转换为模型输入的索引形式
        self.user_to_index = {user: i for i, user in enumerate(self.user_ids)}
        # 正确构建文章id到索引的映射字典，确保不会被意外覆盖
        self.item_to_index = {item: i for i, item in enumerate(self.item_ids)}
        # 处理数据，将用户和文章的对应关系转换为索引对的形式，作为模型最终的输入数据
        self.data = self.process_data()


    def __len__(self):
        # 返回数据集的长度，即数据的条数，也就是用户和文章索引对的数量
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取对应的数据，返回用户和文章的索引张量，用于模型训练等操作
        user_idx, item_idx = self.data[idx]
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(item_idx, dtype=torch.long)

    def get_user_item_time(self, click_df):
        """
        根据点击时间对点击数据进行处理，获取用户的点击文章序列
        :param click_df: 点击数据DataFrame，包含用户、文章以及点击时间等信息
        :return: 用户 - 文章 - 时间序列的字典
        """
        # 按照点击时间对数据进行排序，使得同一用户的点击记录按时间先后顺序排列
        click_df = click_df.sort_values('click_timestamp')

        def make_item_time_pair(df):
            # 将文章id和点击时间组合成元组列表，每个元组表示一次点击的文章和对应的时间
            return list(zip(df['click_article_id'], df['click_timestamp']))

        # 按照用户id进行分组，对每组数据应用make_item_time_pair函数，得到每个用户的文章和时间序列列表
        user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
            lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
        # 将结果转换为字典形式，键为用户id，值为对应的文章和时间序列列表
        return dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    def process_data(self):
        data = []
        # 遍历每个用户及其对应的文章和时间序列列表
        for user, item_time_list in self.user_item_time_dict.items():
            # 获取用户对应的索引
            user_idx = self.user_to_index[user]
            for item, _ in item_time_list:
                # 获取文章对应的索引
                item_idx = self.item_to_index[item]
                # 将用户和文章的索引组合添加到数据列表中，形成最终的模型输入数据格式
                data.append((user_idx, item_idx))
        return data


# 定义推荐模型类，基于神经网络构建，用于学习用户和文章之间的关系以进行推荐
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecommendationModel, self).__init__()
        # 用户嵌入层，将用户id转换为低维向量表示
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # 文章嵌入层，将文章id转换为低维向量表示
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # 全连接层，用于融合用户和文章的向量表示，并输出预测得分
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_idx, item_idx):
        # 获取用户的向量表示
        user_emb = self.user_embedding(user_idx)
        # 获取文章的向量表示
        item_emb = self.item_embedding(item_idx)
        # 将用户和文章的向量在维度1上进行拼接
        combined = torch.cat([user_emb, item_emb], dim=1)
        # 通过全连接层得到预测得分，这里的得分可理解为用户对文章的偏好程度等
        output = self.fc(combined)
        # 去除维度为1的维度，使输出为标量形式
        return output.squeeze()


# 训练模型的函数，按照设定的轮数和数据加载器，进行模型参数的更新优化
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    losses = []  # 用于记录每轮的平均损失值
    accuracies = []  # 新增用于记录每轮准确率值的列表
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_count = 0
        total_count = 0
        # 使用tqdm来可视化每轮训练的数据加载进度，设置总步数为数据加载器的长度
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1} Training', leave=True,
                  bar_format='{l_bar}{bar}|') as pbar:
            for batch_idx, (user_idx, item_idx) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(user_idx, item_idx)
                target = torch.ones_like(output)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # 清理不再使用的张量，释放内存
                del user_idx, item_idx
                # 根据模型输出的得分情况
                predicted = (output > 0.5).float()
                correct_count += (predicted == target).sum().item()
                total_count += len(output)

                # 更新进度条，每处理完一个批次数据，进度条前进一格
                pbar.update(1)

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        epoch_accuracy = correct_count / total_count if total_count > 0 else 0
        accuracies.append(epoch_accuracy)

        print(f'Epoch {epoch + 1} Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')

    return losses, accuracies


# 针对给定用户对所有文章进行评分预测，然后选择得分最高的topk篇文章作为推荐结果
def deep_learning_based_recommend(model, user_idx, all_item_indices, topk):
    model.eval()
    with torch.no_grad():
        scores = []
        count = 0
        selected_indices_set = set()  # 创建集合用于记录已选的文章索引
        for item_idx in all_item_indices:
            score = model(user_idx, item_idx)
            scores.append(score)
            sorted_scores, sorted_indices = torch.sort(torch.tensor(scores), descending=True)
            topk_indices = sorted_indices[:topk].tolist()
            new_count = 0
            for index in topk_indices:
                if index not in selected_indices_set:  # 只统计不在集合中的新索引
                    new_count += 1
                    selected_indices_set.add(index)  # 将新索引添加到集合中
            count += new_count
            if count >= topk:
                break
        return topk_indices

def get_popular_items():
    config = Config()
    all_click = pd.read_csv(config.data_path + 'train_click_log.csv')
    all_click['click_timestamp'] = pd.to_datetime(all_click['click_timestamp'])  # 将时间列转换为日期时间类型
    # 计算相对时间权重，这里简单假设距离当前时间越近权重越高，可根据实际调整权重计算方式
    current_time = datetime.now()
    all_click['time_weight'] = (current_time - all_click['click_timestamp']).dt.days.apply(lambda x: 1 / (x + 1))
    # 过滤掉点击频次过低的文章，这里假设最小点击频次阈值为5
    all_click = all_click[all_click['click_count'] >= 5]
    # 按照文章id分组，计算加权点击频次
    weighted_click_counts = all_click.groupby('click_article_id').apply(lambda x: (x['click_count'] * x['time_weight']).sum()).reset_index()
    weighted_click_counts.columns = ['article_id', 'weighted_click_count']
    # 按照加权点击频次进行降序排序，获取最热门的文章列表
    sorted_items = weighted_click_counts.sort_values(by='weighted_click_count', ascending=False)
    popular_items = sorted_items['article_id'].tolist()
    return popular_items

# 基于文章协同过滤进行召回的函数，根据用户历史点击文章、文章相似性等信息来召回文章
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
    基于文章协同过滤的召回
    :param user_id: 用户id，用于确定召回的目标用户
    :param user_item_time_dict: 字典，根据点击时间获取用户的点击文章序列，格式为 {user1: {item1: time1, item2: time2..}...}
    :param i2i_sim: 字典，文章相似性矩阵，存储文章之间的相似度信息，格式为 {article_id1: {article_id2: similarity_score2,...},...}
    :param sim_item_topk: 整数，选择与当前文章最相似的前k篇文章，用于确定相似文章的筛选数量
    :param recall_item_num: 整数，最后的召回文章数量，即最终要为用户召回多少篇文章
    :param item_topk_click: 列表，点击次数最多的文章列表，用于在召回文章不足时进行补全
    :return: 召回的文章列表，格式为 {item1:score1, item2: score2...}，其中score表示文章的推荐得分等相关信息
    """
    user_hist_items = user_item_time_dict[user_id]

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = -i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


# 获取测试集真实点击数据，整理成 {'user_id': [clicked_article_ids]} 的格式，用于后续计算准确率等评估操作
def get_test_click_data():
    """
    获取测试集真实点击数据，整理成 {'user_id': [clicked_article_ids]} 的格式
    :return: 测试集真实点击数据字典，键为用户id，值为该用户在测试集中点击过的文章id列表
    """
    tst_click = pd.read_csv(Config().data_path + 'testA_click_log.csv')
    # 按照用户分组，将每个用户的点击文章id转换为列表，并转换为字典形式
    click_data = tst_click.groupby('user_id')['click_article_id'].apply(list).to_dict()
    return click_data


# 计算准确率的函数，通过对比推荐文章是否在测试集用户真实点击文章列表里来计算准确率
accuracies = []  # 新增一个列表，用于记录每轮的准确率值


def calculate_accuracy(tst_recall):
    global accuracies
    correct_count = 0
    total_count = 0
    tst_click_data = get_test_click_data()
    for index, row in tst_recall.iterrows():
        user_id = row['user_id']
        recommended_article_id = row['click_article_id']
        if recommended_article_id in tst_click_data.get(user_id, []):
            correct_count += 1
        total_count += 1
    acc = correct_count / total_count if total_count > 0 else 0
    accuracies.append(acc)
    return acc

def submit(recall_df, topk=5, model_name=None):
    """
    :param recall_df: 召回结果数据DataFrame，包含用户id、推荐文章id以及预测得分等信息
    :param topk: 每个用户最终提交的文章数量，即要筛选出每个用户得分最高的topk篇文章用于提交
    :param model_name: 模型名称，用于生成提交文件的文件名，方便区分不同模型的提交结果
    """
    # 先备份一份原始的带有pred_score列的数据
    recall_df_with_score = recall_df.copy()
    # 按照用户id和预测得分对召回数据进行排序，以便后续进行排名等操作
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    # 为每个用户的召回文章计算排名，排名依据是预测得分降序排列，用于确定每个用户推荐文章的先后顺序
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    # 判断是不是每个用户都有5篇文章及以上，若不足则进行补全
    tmp = recall_df.groupby('user_id')['rank'].apply(lambda x: x.max())
    for user in tmp[tmp < topk].index:
        # 获取热门文章列表
        popular_items = get_popular_items()
        existing_items = recall_df[recall_df['user_id'] == user]['click_article_id'].tolist()
        # 统一文章id的数据类型，确保比较准确
        existing_items = [str(item) for item in existing_items]
        # 计算还需要补全的文章数量
        num_to_fill = topk - len(existing_items)
        filled_items = [item for item in popular_items if str(item) not in existing_items][:num_to_fill]
        new_rows = []
        for item in filled_items:
            new_rows.append({'user_id': user, 'click_article_id': item, 'pred_score': 0})
        recall_df = pd.concat([recall_df, pd.DataFrame(new_rows)], ignore_index=True)
        recall_df = recall_df.reset_index(drop=True)  # 重置索引
        # 验证补全后每个用户的文章列表是否存在重复文章
        user_df = recall_df[recall_df['user_id'] == user]
        unique_article_ids = set(user_df['click_article_id'].tolist())
        if len(unique_article_ids) < len(user_df):
            print(f"Warning: User {user} has duplicate articles after filling.")
        else:
            print(f"User {user} has been filled successfully with {len(filled_items)} items.")
        # 进一步验证补全后每个用户的文章数量是否确实达到预期
        filled_count = len(user_df)
        if filled_count < topk:
            print(f"Warning: User {user} still has less than {topk} articles after filling.")
        print(
            f"User {user} has been filled with {len(filled_items)} items. Current recall_df shape: {recall_df.shape}")
    # 删除召回数据中的预测得分列，因为在提交文件中不需要该列信息，按照要求只保留用户和推荐文章相关信息
    del recall_df['pred_score']
    # 筛选出每个用户排名在前topk的文章，并将数据进行透视操作，将user_id作为索引，rank作为列，click_article_id作为值
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名，使生成的提交文件符合要求的格式规范
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = Config().save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)



if __name__ == "__main__":
    config = Config()
    all_click_df = get_all_click_df()

    dataset = ClickDataset(all_click_df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_users = len(dataset.user_ids)
    num_items = len(dataset.item_ids)
    print(num_users)
    embedding_dim = 32
    model = RecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    losses, accuracies = train(model, dataloader, criterion, optimizer, num_epochs)  # 获取损失值和准确率值列表

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.show()

    topk = 10
    all_item_indices = torch.arange(len(dataset.item_ids)).unsqueeze(1).to(torch.long)
    user_recall_items_dict = collections.defaultdict(list)
    for user_idx in range(len(dataset.user_ids)):
        topk_indices = deep_learning_based_recommend(model, torch.tensor([user_idx], dtype=torch.long),
                                                     all_item_indices, topk)
        topk_items = [dataset.item_ids[i] for i in topk_indices]
        user_recall_items_dict[user_idx] = topk_items

    user_item_score_list = []
    for user, items in user_recall_items_dict.items():
        for item in items:
            user_item_score_list.append([user, item, 0])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

    tst_click = pd.read_csv(config.data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

    submit(tst_recall, topk=5, model_name='deep_learning_baseline')

