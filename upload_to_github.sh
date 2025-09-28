#!/bin/bash

# GitHub Upload Helper Script
# This script helps you upload your ASL Gesture Detection project to GitHub

echo "🤟 ASL Gesture Detection - GitHub Upload Helper"
echo "================================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please run 'git init' first."
    exit 1
fi

echo "✅ Git repository found"
echo ""

# Get repository details from user
echo "📝 Please provide your GitHub repository details:"
echo ""
read -p "GitHub Username: " GITHUB_USERNAME
read -p "Repository Name: " REPO_NAME

if [ -z "$GITHUB_USERNAME" ] || [ -z "$REPO_NAME" ]; then
    echo "❌ Username and repository name are required!"
    exit 1
fi

echo ""
echo "🔗 Setting up remote repository..."
echo "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo ""

# Add remote origin
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Set main branch
git branch -M main

echo "📤 Pushing to GitHub..."
echo ""

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your project has been uploaded to GitHub!"
    echo "🌐 View your repository at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "📋 Next steps:"
    echo "1. Visit your repository on GitHub"
    echo "2. Add a description and topics"
    echo "3. Enable GitHub Pages if desired"
    echo "4. Share your awesome ASL detection project!"
else
    echo ""
    echo "❌ Push failed. Please check:"
    echo "1. Repository exists on GitHub"
    echo "2. You have write permissions"
    echo "3. Your GitHub credentials are correct"
    echo ""
    echo "💡 You can also push manually:"
    echo "   git push -u origin main"
fi
