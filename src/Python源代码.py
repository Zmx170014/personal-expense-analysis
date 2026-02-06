# ========== 导入库 ==========
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# ========== 中文显示配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#999999']

# ========== 数据加载 ==========
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='gbk', skiprows=10, header=0)
    return df

file_path = r'C:\Users\邹顺羽\Desktop\cashbook_record_20260202_104516.csv'
df = load_data(file_path)

# ====================== 数据加载后探查 ======================
print("=" * 60)
print("📊 原始数据概览")
print("=" * 60)
print(f"原始数据形状：{df.shape}")
print("\n原始数据前5行：")
print(df.head())
print("\n原始数据列名：")
print(df.columns.tolist())
print("\n原始数据缺失值统计：")
print(df.isnull().sum())
print("\n原始数据重复行数量：", df.duplicated().sum())

# ========== 数据清洗（分步展示） ==========
print("\n" + "=" * 60)
print("🧹 开始数据清洗...")
print("=" * 60)

# 1. 删除无用列
print("\n【步骤1】删除无用列 'Unnamed: 8'")
df = df.drop(columns=['Unnamed: 8'], errors='ignore')
print(f"删除后列数：{len(df.columns)}，剩余列：{df.columns.tolist()}")

# 2. 筛选支出数据
print("\n【步骤2】筛选「收支类型 = 支出」的数据")
expense_df = df[df['收支类型'] == '支出'].copy().reset_index(drop=True)
print(f"筛选后数据量：{expense_df.shape[0]} 行（原 {df.shape[0]} 行）")

# 3. 清洗金额字段
print("\n【步骤3】清洗「金额」字段（去除非数字字符）")
def clean_amount(s):
    if pd.isna(s):
        return 0.0
    return float(re.sub(r'[^0-9.]', '', str(s)))

expense_df['金额'] = expense_df['金额'].apply(clean_amount).astype(float)
expense_df = expense_df[expense_df['金额'] > 0]
print(f"过滤金额≤0后，剩余数据量：{expense_df.shape[0]} 行")

# 4. 转换时间字段
print("\n【步骤4】转换「记录时间」为日期格式")
expense_df['记录时间'] = pd.to_datetime(expense_df['记录时间'], errors='coerce')
expense_df = expense_df.dropna(subset=['记录时间'])  # 删除时间缺失行
expense_df['年'] = expense_df['记录时间'].dt.year
expense_df['月'] = expense_df['记录时间'].dt.month
expense_df['日'] = expense_df['记录时间'].dt.day  # 新增日维度，用于日度折线图
expense_df['星期'] = expense_df['记录时间'].dt.weekday + 1

# 5. 分类名称统一
print("\n【步骤5】统一「分类」名称")
expense_df['分类'] = expense_df['分类'].replace({
    '餐饮': '饮食', '交通': '出行', '生活日用': '日用品',
    '休闲玩乐': '娱乐', '服饰鞋包': '购物'
})

# ====================== 清洗完成后验证 ======================
print("\n" + "=" * 60)
print("✅ 数据清洗完成，最终数据概览")
print("=" * 60)
print(f"最终数据形状：{expense_df.shape}")
print("\n最终数据前5行：")
print(expense_df[['记录时间', '分类', '金额', '年', '月', '日']].head())

# ========== 数据分析 ==========
category_total = expense_df.groupby('分类')['金额'].sum().sort_values(ascending=False)
total_expense = category_total.sum()


# 3. 按金额降序排序，取前10条
top10_expense = df.sort_values(by="金额", ascending=False).head(10)

# 4. 打印结果（展示关键列）
print("金额最大的10笔支出：")
print(top10_expense[["记录时间", "分类", "金额", "备注"]].to_string(index=False))

# 合并小分类（占比<5%）
category_pct = category_total / total_expense * 100
small_categories = category_pct[category_pct < 5].index
if len(small_categories) > 0:
    other_amount = category_total[small_categories].sum()
    category_total = category_total.drop(small_categories)
    category_total['其他'] = other_amount
    category_total = category_total.sort_values(ascending=False)

# 时间维度统计
month_expense = expense_df.groupby('月')['金额'].sum()  # 月度支出（折线图核心数据）
day_expense = expense_df.groupby('记录时间')['金额'].sum()  # 日度支出（新增，按日期聚合）
week_expense = expense_df.groupby('星期')['金额'].sum()

# ========== 可视化1：消费结构饼图（保留原有） ==========
fig1, ax1 = plt.subplots(figsize=(10, 10))
explode = [0.05 if i == 0 else 0 for i in range(len(category_total))]
wedges, _, autotexts = ax1.pie(
    category_total.values, autopct='%1.1f%%', colors=colors, startangle=90,
    explode=explode, textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)
ax1.legend(wedges, category_total.index, title="消费类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), prop={'size': 10})
ax1.set_title(f'消费类别占比（总支出：{total_expense:.2f}元）', fontsize=14)
ax1.axis('equal')
plt.tight_layout()
plt.savefig('消费结构分析_饼图.png', dpi=300)
plt.show()

# ========== 可视化2：各分类总支出柱形图（保留原有） ==========
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars1 = ax2.bar(category_total.index, category_total.values, color=colors[:len(category_total)])
ax2.set_title('各分类总支出对比', fontsize=14)
ax2.set_ylabel('支出金额（元）', fontsize=12)
ax2.tick_params(axis='x', rotation=30, labelsize=10)
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2, height+50, f'{height:.0f}', ha='center', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('各分类总支出对比.png', dpi=300)
plt.show()

# ========== 可视化3：【新增】月度支出趋势折线图（时间维度） ==========
fig3, ax3 = plt.subplots(figsize=(12, 6))
# 绘制折线图，添加标记点、线条优化
ax3.plot(month_expense.index, month_expense.values, color='#45B7D1', linewidth=2.5, marker='o', markersize=8, markerfacecolor='white', markeredgecolor='#45B7D1', markeredgewidth=2)
# 添加数值标注
for x, y in zip(month_expense.index, month_expense.values):
    ax3.text(x, y+100, f'{y:.0f}', ha='center', va='bottom', fontsize=10, weight='bold')
# 图表美化
ax3.set_title('月度支出趋势折线图（1-12月）', fontsize=14, pad=15)
ax3.set_xlabel('月份', fontsize=12)
ax3.set_ylabel('支出金额（元）', fontsize=12)
ax3.set_xticks(range(1, 13))  # 确保x轴显示1-12月
ax3.grid(True, alpha=0.3, linestyle='--')  # 添加网格，更清晰看趋势
plt.tight_layout()
plt.savefig('月度支出趋势_折线图.png', dpi=300)
plt.show()

# ========== 可视化4：【新增】日度支出趋势折线图（更细粒度时间维度） ==========
fig4, ax4 = plt.subplots(figsize=(14, 6))
# 绘制日度折线图
ax4.plot(day_expense.index, day_expense.values, color='#FF6B6B', linewidth=2, alpha=0.8)
# 图表美化
ax4.set_title('日度支出趋势折线图', fontsize=14, pad=15)
ax4.set_xlabel('日期', fontsize=12)
ax4.set_ylabel('支出金额（元）', fontsize=12)
ax4.tick_params(axis='x', rotation=45, labelsize=10)  # 日期标签旋转，避免重叠
ax4.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('日度支出趋势_折线图.png', dpi=300)
plt.show()

# ========== 可视化5：星期消费规律柱形图（保留原有） ==========
fig5, ax5 = plt.subplots(figsize=(10, 6))
week_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
bars3 = ax5.bar(week_labels, week_expense.reindex(range(1,8)).values, color=colors[2])
ax5.set_title('星期消费规律', fontsize=14)
ax5.set_ylabel('支出金额（元）', fontsize=12)
ax5.tick_params(axis='x', labelsize=10)
for bar in bars3:
    height = bar.get_height()
    ax5.text(bar.get_x()+bar.get_width()/2, height+20, f'{height:.0f}', ha='center', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('星期消费规律.png', dpi=300)
plt.show()

print("\n✅ 所有图表生成完成！1张饼图 + 2张柱形图 + 2张时间维度折线图，已分别保存")

