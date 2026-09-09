cat > README.md <<'EOF'
# Mechanistic Interpretability Research

Research workspace for mechanistic interpretability experiments.

## Local workflow

Develop experiments locally in Cursor.

Commit important code to GitHub.

## Vast workflow

1. Rent GPU.
2. Configure `vast-mech` in `~/.ssh/config`.
3. Sync code:

   ./scripts/sync_to_vast.sh

4. SSH:

   ssh vast-mech

5. Bootstrap:

   cd /workspace/mech-interp
   ./scripts/bootstrap_vast.sh

6. Run experiment:

   ./scripts/start_experiment.sh experiments/gpu_smoke_test.py

7. Download results:

   ./scripts/sync_from_vast.sh

8. Stop/destroy Vast instance after results are safe.
EOF