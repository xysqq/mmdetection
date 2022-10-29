from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SonicAfterRunHook(Hook):

    def after_run(self, runner):
        runner.after_run()
