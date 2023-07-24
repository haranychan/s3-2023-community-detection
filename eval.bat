
for /l %%n in (1,1,31) do (
    C:\Git\04_Public\s3-2023-community-detection\src\evaluators\community_detection.py --input-file C:\Git\04_Public\s3-2023-community-detection\data\community-detection\regnier300-50.txt --output-file C:\Git\04_Public\s3-2023-community-detection\output\community-detection\regnier300-50_grasp_a09_%%n.txt
)