import os
import sys

if __name__ == "__main__":
    '''
    检查文件是否存在
    '''
    if not (os.path.exists("misc.txt") and os.path.exists("houror.txt")):
        print("Required files are missing. Please ensure 'misc.txt' and 'houror.txt' exist in the current directory.")
        sys.exit(1)
    if os.path.exists("label.txt"):
        os.remove("label.txt")
    
    with open("label.txt", 'a+', encoding='utf-8') as new_file:
        with open("misc.txt", 'r', encoding='utf-8') as file:
            while line := file.readline():
                new_file.write('daily\t' + line)
            file.close()
        with open("houror.txt", 'r', encoding='utf-8') as file:
            while line := file.readline():
                new_file.write('wink\t' + line)
            file.close()
        new_file.close()
    print("Rewrite completed successfully.")
        
            