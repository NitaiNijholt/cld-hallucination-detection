        except Exception as e:
            logger.error(f"❌ CI metrics computation failed: {e}")
            logger.exception("Full traceback for CI metrics failure:")
            edge_ci_metrics = {}
    else:
        edge_ci_metrics = {}

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
