from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="llSourcell/doctorGPT_mini")

output = cm.generate(
    prompt='I lost my inhaler. What do I do?',
    progress_callback=StreamToStdout(callback_interval=2)
)
