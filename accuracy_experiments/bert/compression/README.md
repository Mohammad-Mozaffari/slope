## Instruction on Setting Up This Repository as Subtree

In your main repo, add the current `compression` repository and pull as a subtree.

```bash
git remote add compression https://github.com/Mohammad-Mozaffari/compression.git
git subtree pull --prefix=compression/ compression main
```

## Instruction on Making Changes on Main Repository and This Repository

Commit your changes first, then push to current `compression` repository.
Next, pull the changes to your main repository to reflect the changes.
Finally, push the reflected changes in `compression` to your main repository.

```bash
git subtree push --prefix=compression/ compression main
git subtree pull --prefix=compression/ compression main
git push
```