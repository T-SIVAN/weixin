"""Repository marker for test discovery.

The application package lives in :mod:`weixin_lite`.  Keeping this module
side-effect free prevents pytest from importing the repository directory as a
package and looking for non-existent root-level modules.
"""
