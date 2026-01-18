# 使用方法：bash ./wyren/scripts/grep.sh ./EasyEdit(要搜索的文件夹路径) portability(要搜索的关键词)
#!/bin/bash

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "Usage: $0 <search_path> <search_term>"
    echo "Example: $0 ./EasyEdit portability"
    exit 1
fi

# 获取参数
SEARCH_PATH="$1"
SEARCH_TERM="$2"

# 获取当前工作目录
CURRENT_DIR="$(pwd)"
echo "当前工作目录: $CURRENT_DIR"

# 创建输出目录
BASENAME=$(basename "$SEARCH_PATH")
OUTPUT_DIR="$CURRENT_DIR/wyren/2grep/$BASENAME"
mkdir -p "$OUTPUT_DIR"

# 设置输出文件路径
OUTPUT_FILE="$OUTPUT_DIR/${SEARCH_TERM}.log"

# 构建并显示grep命令
# GREP_CMD="111grep -rn \"$SEARCH_TERM\" \"$SEARCH_PATH/\""
GREP_CMD="grep -rn \"$SEARCH_TERM\" \"$SEARCH_PATH/\" 2>/dev/null | while read line; do echo \"\$(pwd)\${line#.}\"; done"

# echo "执行的 grep 命令: $GREP_CMD"
# echo ""

# 写入命令信息到文件
echo "# grep 命令: $GREP_CMD" > "$OUTPUT_FILE"
echo "# 执行时间: $(date)" >> "$OUTPUT_FILE"
echo "# 工作目录: $CURRENT_DIR" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 显示搜索结果标题
echo "搜索结果:"
echo "----------------------------------------"

# 创建临时文件用于计数
TEMP_FILE=$(mktemp)

# 执行搜索并处理输出
# 使用 tee 命令同时：1) 显示在终端 2) 保存到临时文件
grep -rn "$SEARCH_TERM" "$SEARCH_PATH/" 2>/dev/null | while read line; do 
    echo "$CURRENT_DIR${line#.}"
done | tee "$TEMP_FILE"  # 同时输出到终端和临时文件

# 将搜索结果追加到输出文件
cat "$TEMP_FILE" >> "$OUTPUT_FILE"

echo "搜索完成！结果已保存到: $OUTPUT_FILE"
echo "当前目录: $CURRENT_DIR"
