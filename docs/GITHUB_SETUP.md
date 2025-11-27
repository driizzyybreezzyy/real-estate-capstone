# How to Push Your Project to GitHub

Since `git` is not currently installed on your system, you will need to install it first. Follow these steps to preserve your work on GitHub.

## Step 1: Install Git
1.  Download Git for Windows: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2.  Run the installer and accept the default settings.
3.  **Restart your terminal/computer** after installation.

## Step 2: Create a Repository on GitHub
1.  Log in to [GitHub.com](https://github.com).
2.  Click the **+** icon in the top-right corner and select **New repository**.
3.  Name it `real-estate-capstone`.
4.  **Do NOT** check "Add a README", ".gitignore", or "license" (we already have these locally).
5.  Click **Create repository**.

## Step 3: Push Your Code
Open your terminal (PowerShell or Command Prompt) in the `real_estate_capstone` folder and run these commands one by one:

```bash
# 1. Initialize Git
git init

# 2. Add all files (the .gitignore file I created will handle exclusions)
git add .

# 3. Commit your work
git commit -m "Phase 1 Complete: Professional Real Estate Capstone Pipeline"

# 4. Link to your GitHub repo (Replace URL with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/real-estate-capstone.git

# 5. Push the code
git branch -M main
git push -u origin main
```

## Backup
I have also created a local zip backup for you: `real_estate_capstone_backup.zip`. You can keep this safe just in case!
