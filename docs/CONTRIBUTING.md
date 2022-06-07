# Contributing Guidelines

We accept different types of contributions,
including some that don't require you to write a single line of code.

## 📝 Types of contributions

### Discussions 🎉

Discussions are where we have conversations.

If have a great new idea, or want to share something amazing with the community,
join us in [discussions](https://github.com/weiji14/zen3geo/discussions).

### Issues 🐞

[Issues](https://docs.github.com/en/github/managing-your-work-on-github/about-issues)
are used to track tasks that contributors can help with.

If you've found something in the content or the website that should be updated,
search open issues to see if someone else has reported the same thing. If it's
something new, [open an issue](https://github.com/weiji14/zen3geo/issues/new/choose)!
We'll use the issue to have a conversation about the problem you want to fix.

### Pull requests 🛠️

A [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
is a way to suggest changes in our repository.

When we merge those changes, they should be deployed to the live site within a few minutes.
To learn more about opening a pull request in this repo,
see [Opening a pull request](#opening-a-pull-request) below.

### Translations 🌏

人虽有南北之分，但佛性本无南北。

Yes, the source content in this repository is mostly written in English,
but we welcome folks from across the world! Please reach out if you have experience in translations and are interested in contributing!

---

## 👐 Opening a Pull Request

1. [Login](https://github.com/login) to your GitHub account,
    or sign up for a new one at https://github.com/signup.

2. Navigate to the file you want to modify, e.g. the
   [API docs file](https://github.com/weiji14/zen3geo/blob/main/docs/api.md).

3. Click on the pen 🖊️ icon on the top right corner that says "Edit this file"

4. This should bring you to a page similar to
   https://github.com/weiji14/zen3geo/edit/main/docs/api.md
   where you can make edits to the text using a web-based editor.
   Feel free to switch between the "Edit file" and "Preview changes" tabs as
   you modify the content to make sure things look ok.

5. Once you're happy with your changes, scroll down to the bottom where it says
   **Commit changes**. This is where you will add a short summary of the
   changes you have made.

   ![The place to commit changes](https://user-images.githubusercontent.com/23487320/172029885-947e4e24-675a-4498-a2d8-f1fa4c26b934.png)

   Specifically, in the first box, you will need to give a short title (e.g.
   "Fixed typo in api.md file") that describes the changes you've made.
   Optionally, you can write a few extra sentences in the second box to explain
   things in more detail.

6. Select the "Create a new branch for this commit and start a pull request"
   option and provide a new branch name (e.g. "fix-api-typo"). What this
   does is to ensure your changes are made in an independent manner or 'branch'
   away from the main trunk, and those changes will have the opportunity to be
   double checked and openly reviewed by other people.

7. Click on the green 'Propose changes' button. This will bring you to a new
   page.

8. Almost there! This "Open a pull request" page is where you can finalize
   things for the 'pull request' (a request to make changes) you will be
   opening soon. Again you will need to provide a title (e.g. 'Minor changes to
   the API markdown file') and a description.

   ![Pull request dialog page](https://user-images.githubusercontent.com/23487320/172030066-63dbdaa3-c7d4-403f-a3b6-5bccd966d038.png)

   Be sure to provide any context on **why** you are making the changes, and
   **how** you are doing so. This will make it easier for other people to
   know what is happening when they review your changes.

9. Ready? Click on the green 'Create pull request' button! This will make your
   changes available for everyone to see and review publicly. The maintainers
   will be notified about your great new addition and will get back to you on
   the next steps.

---

## 🏠 Running things locally

This project uses [``poetry``](https://python-poetry.org/docs/master/) for
installing Python dependencies required in ``zen3geo``, as well as the
development and documentation-related dependencies.

### Cloning the repository ♊

```
git clone git@github.com:weiji14/zen3geo.git
cd zen3geo
```

### Setup virtual environment ☁️

```
mamba create --name zen3geo python=3.9
mamba activate

pip install poetry==1.2.0b1
poetry install
```

### Building documentation 📖

```
poetry install --extras=docs  # or `pip install .[docs]`
jupyter-book build docs/
```

Then open ``docs/_build/html/index.html`` in your browser to see the docs.

---

## 🥳 And that's it!

You're now part of the zen3geo community ✨

```{admonition} Credits
:class: seealso
*This contributing guide was adapted from*
[GitHub docs](https://github.com/github/docs/blob/main/contributing/types-of-contributions.md)
and the [APECS-Earth-Observation/Polar-EO-Database](https://github.com/APECS-Earth-Observation/Polar-EO-Database/blob/main/CONTRIBUTING.md) project.
```
