# coding UTF-8


# start Fiji script
def call_it(path):
    import subprocess
    #subprocess.Popen(['/home/pascal/computer-zeugs/Git/sted_automation/Util/script2'])
    subprocess.getoutput([path])
call_it('./call-fiji-bashscript')

