import dashscope
dashscope.api_key=""
from http import HTTPStatus

from http import HTTPStatus
from dashscope import MultiModalConversation


def conversation_call(filepath="./", prompt="mouse"):
    """Sample of multiple rounds of conversation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": 'E:\\pythonProject\\VLM\\OIP-C.jpg'},
                {"text": "What did you see on this picture?"},
            ]
        }
    ]
    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.
    messages.append({'role': response.output.choices[0].message.role,
                     'content': [{'text': response.output.choices[0].message.content}]})
    messages.append({"role": "user",
                     "content": [
                         {"text": "输出'mouse'检测框", }
                     ]})

    response = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                           messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.


if __name__ == '__main__':
    conversation_call()