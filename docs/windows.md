# Windows Specific Setting

## Symbolic Link

This project uses symbolic links to point to the stable version of prompts used for each task.
While symbolic links are supported on Linux and MacOS, they are not supported on Windows.
If you are on Windows, the following files will be plain text files instead of symbolic links:
`assets/prompts/CP/stable`, `assets/prompts/ED/stable`, `assets/prompts/IE/stable`

They are plain text files that contain the path to the stable version of prompts used for each task.

You can manually copy the target directory to `assets/prompts/CP/stable`, `assets/prompts/ED/stable`, `assets/prompts/IE/stable` to make the code work.




