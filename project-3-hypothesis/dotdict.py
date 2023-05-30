class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def pick(self, *keys):
    return dotdict({key: self[key] for key in keys})
