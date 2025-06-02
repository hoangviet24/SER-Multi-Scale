import tkinter as tk

root = tk.Tk()

# Create the Text widget
text_widget = tk.Text(root)
text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create the scrollbar
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=text_widget.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the Text widget to use the scrollbar
text_widget.configure(yscrollcommand=scrollbar.set)

# Insert some text to test scrolling
for i in range(50):
    text_widget.insert(tk.END, f"Line {i+1}\n")
button = tk.Button(root, text="Exit", command=root.quit)
button.pack(side=tk.BOTTOM)
root.mainloop()