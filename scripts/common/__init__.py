"""Infrastructure shared by every training script: arguments, data, model setup, metrics.

Nothing here is specific to a method. What distinguishes one method from another -- how it turns
the two views of an unlabeled batch into a loss -- stays in that method's own file, so that
reading fixmatch.py next to efficientmatch.py shows the whole difference between them.
"""
