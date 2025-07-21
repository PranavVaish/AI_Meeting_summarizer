from gtts import gTTS

speech_text = """Good morning, everyone.
Today, I want to remind us all about the importance of human rights—the basic freedoms and protections that belong to every one of us.

First, human rights are universal. No matter where we are from, what we believe, or how we look, we all have the right to live with dignity and respect.
Second, they ensure equality. Human rights protect us from discrimination, promoting fairness regardless of gender, race, religion, or background.
Third, they safeguard our freedom of expression, allowing us to speak, think, and share ideas without fear.
Fourth, human rights guarantee access to justice—ensuring that no one is above the law and everyone deserves protection under it.
And fifth, they promote peace and development. Where human rights are respected, societies thrive and conflicts reduce.

Let's not take them for granted. Let's stand up, speak out, and protect the rights of every human being.

Thank you.
"""

tts = gTTS(text=speech_text, lang='en')
tts.save("human_rights_speech.mp3")
print("Audio saved as human_rights_speech.mp3")
