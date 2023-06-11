import gradio as gr
import multi_detect
import bm25

file_paths = []

def upload_file(files):
    for file in files:
        file_paths.append(file.name)
    # print(file_paths)
    return file_paths

def solution(text_input):
    lst = multi_detect.multi_dectect(file_paths)
    lst_prority = bm25.run_bm25(lst, text_input)
    output_path = [file_paths[i] for i in lst_prority]
    return output_path

def clear_file():
    file_paths.clear()
    return file_paths

with gr.Blocks() as program:
    #Tiêu đề
    gr.Markdown("Tìm kiếm từ khóa trong một tập hình ảnh")
    #Nơi hiển thị các file được upload
    file_upload = gr.File(label="File ảnh cần tìm kiếm")
    #Tạo button upload file va clear các file 
    with gr.Row():
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["image"], file_count="multiple")
        clear_button = gr.Button("Xóa các file đã thêm")
        clear_button.click(fn=clear_file, outputs=file_upload)
    #listener event upload_button
    upload_button.upload(fn=upload_file, inputs=upload_button, outputs=file_upload)
    #textbox get keyword
    text_input = gr.Textbox(placeholder="Nhập từ khóa muốn tìm kiếm", label="Từ khóa cần tìm")
    #file result
    file_result = gr.File(label="File ảnh kết quả", type="binary")
    #Button submit
    submit = gr.Button("Tìm kiếm")
    #listener event button submit
    submit.click(fn=solution, inputs=text_input, outputs=file_result)
       
if __name__ == '__main__' :
    program.launch(share=True, inbrowser=True)