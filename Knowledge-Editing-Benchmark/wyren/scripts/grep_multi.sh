# 使用方法：bash ./wyren/scripts/grep_multi.sh ./EasyEdit(要搜索的文件夹路径) locality portability (要搜索的关键词)
# #!/bin/bash

# # 检查参数
# if [ $# -lt 2 ]; then
#     echo "用法: $0 <搜索路径> <关键词1> [关键词2] [关键词3] ..."
#     exit 1
# fi

# SEARCH_PATH="$1"
# shift
# WORDS=("$@")
# COUNT=${#WORDS[@]}

# CURRENT_DIR="$(pwd)"
# OUTPUT_DIR="$CURRENT_DIR/wyren/2grep"
# mkdir -p "$OUTPUT_DIR"

# # 生成输出文件名
# if [ $COUNT -eq 1 ]; then
#     OUTPUT_FILE="$OUTPUT_DIR/${WORDS[0]}.log"
# else
#     FILENAME=$(IFS=_ ; echo "${WORDS[*]}").log
#     OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
# fi

# echo "搜索: ${WORDS[*]}"
# echo "路径: $SEARCH_PATH"
# echo ""

# # 构建搜索命令
# CMD="grep -rl '${WORDS[0]}' '$SEARCH_PATH' 2>/dev/null"
# for (( i=1; i<COUNT; i++ )); do
#     CMD="$CMD | xargs grep -l '${WORDS[i]}' 2>/dev/null"
# done

# # 执行并显示
# echo "找到的文件:"
# echo "------------------------"
# eval "$CMD" | tee "$OUTPUT_FILE"
# echo "------------------------"

# # 统计
# FOUND=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
# echo "共找到 $FOUND 个文件"
# echo "结果保存: $OUTPUT_FILE"

#!/bin/bash

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <搜索路径> <关键词1> [关键词2] [关键词3] ..."
    exit 1
fi

SEARCH_PATH="$1"
shift
WORDS=("$@")
COUNT=${#WORDS[@]}


BASENAME=$(basename "$SEARCH_PATH")
CURRENT_DIR="$(pwd)"
OUTPUT_DIR="$CURRENT_DIR/wyren/2grep/$BASENAME"
mkdir -p "$OUTPUT_DIR"

# 生成输出文件名
if [ $COUNT -eq 1 ]; then
    OUTPUT_FILE="$OUTPUT_DIR/${WORDS[0]}.log"
else
    FILENAME=$(IFS=_ ; echo "${WORDS[*]}").log
    OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
fi



# 获取搜索路径的绝对路径
if [[ "$SEARCH_PATH" = /* ]]; then
    ABS_SEARCH_PATH="$SEARCH_PATH"
else
    ABS_SEARCH_PATH="$(cd "$SEARCH_PATH" && pwd)"
fi
echo "搜索: ${WORDS[*]}"
echo "路径: $ABS_SEARCH_PATH"
echo ""

# 写入信息到文件
echo "# 执行时间: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 显示搜索结果标题
echo "搜索结果:"
echo "----------------------------------------"

# 构建搜索命令
CMD="grep -rl '${WORDS[0]}' '$ABS_SEARCH_PATH' 2>/dev/null"
for (( i=1; i<COUNT; i++ )); do
    CMD="$CMD | xargs grep -l '${WORDS[i]}' 2>/dev/null"
done

# 执行并显示

# 方法1：直接使用 find 和 realpath（最简单）
find "$ABS_SEARCH_PATH" -type f 2>/dev/null | while read file; do
    # 检查文件是否包含所有关键词
    CONTAINS_ALL=true
    for word in "${WORDS[@]}"; do
        if ! grep -q "$word" "$file" 2>/dev/null; then
            CONTAINS_ALL=false
            break
        fi
    done
    
    if [ "$CONTAINS_ALL" = true ]; then
        # 获取绝对路径
        abs_path="$(realpath "$file")"
        echo "$abs_path"
    fi
done | tee "$OUTPUT_FILE"

echo "------------------------"

# 统计
FOUND=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
echo "共找到 $FOUND 个文件"
echo "结果保存: $OUTPUT_FILE"