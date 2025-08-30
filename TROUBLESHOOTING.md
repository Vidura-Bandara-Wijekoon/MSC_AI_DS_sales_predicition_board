# Deployment Troubleshooting Guide

## Python Installation Issues

### Problem: "invalid gzip header" or "failed to extract tar"
**Platforms affected:** Render, some shared hosting

**Root cause:** Platform trying to install Python 3.8.20 from corrupted or incompatible archive

**Solutions:**
1. **Use Docker deployment** (recommended)
   - Bypasses Python compilation entirely
   - Uses pre-built, tested Python environments
   - Works on all major platforms

2. **Switch hosting platforms**
   - Railway: Excellent Docker support
   - Fly.io: Reliable Python environments
   - DigitalOcean: Enterprise-grade infrastructure

3. **Force cache clearing**
   - Clear build cache on platform
   - Delete and recreate deployment
   - Use manual deployment triggers

### Problem: Python version mismatch
**Symptoms:** Platform installs wrong Python version

**Solutions:**
1. Use `runtime.txt` with exact version: `python-3.11.0`
2. Specify in platform config (render.yaml, etc.)
3. Use Dockerfile with explicit Python version

## Docker Issues

### Problem: Docker build fails
**Common causes:**
- Missing Dockerfile
- Invalid requirements.txt
- Network issues during pip install

**Solutions:**
1. Verify Dockerfile exists and is valid
2. Test requirements.txt locally: `pip install -r requirements.txt`
3. Add retry logic to pip install
4. Use specific package versions

### Problem: Container starts but app doesn't respond
**Symptoms:** Build succeeds but health checks fail

**Solutions:**
1. Check port configuration (use $PORT environment variable)
2. Verify app binds to 0.0.0.0, not localhost
3. Check application logs for startup errors
4. Ensure all dependencies are installed

## Platform-Specific Issues

### Render
**Common issues:**
- Python compilation failures
- Cache not clearing properly
- Environment variable issues

**Solutions:**
1. Use Docker deployment (env: docker)
2. Clear build cache manually
3. Check service logs for detailed errors
4. Verify render.yaml syntax

### Railway
**Common issues:**
- Port configuration
- Environment variable setup

**Solutions:**
1. Use $PORT environment variable
2. Check Railway dashboard for logs
3. Verify GitHub integration

### Heroku
**Common issues:**
- Procfile configuration
- Buildpack selection
- Dyno sleeping

**Solutions:**
1. Use container stack: `heroku stack:set container`
2. Verify Procfile syntax
3. Check dyno status

## General Debugging Steps

### 1. Check Build Logs
- Look for specific error messages
- Identify which step fails
- Check for network/timeout issues

### 2. Verify Configuration Files
- Dockerfile syntax
- requirements.txt validity
- Platform-specific configs

### 3. Test Locally
```bash
# Test Docker build
docker build -t sales-dashboard .
docker run -p 8080:8080 sales-dashboard

# Test Python environment
pip install -r requirements.txt
cd app && python flask_app.py
```

### 4. Check Dependencies
- Verify all packages in requirements.txt exist
- Check for version conflicts
- Test with minimal requirements first

## Quick Fixes Checklist

- [ ] Clear platform build cache
- [ ] Verify all config files are committed
- [ ] Check Python version specification
- [ ] Test Docker build locally
- [ ] Verify port configuration
- [ ] Check environment variables
- [ ] Review platform-specific logs
- [ ] Try manual deployment
- [ ] Switch to alternative platform

## Emergency Deployment

If all else fails, use this minimal approach:

1. **Create new repository** with just essential files
2. **Use Railway** for fastest deployment
3. **Minimal requirements.txt** with only essential packages
4. **Simple Dockerfile** based on python:3.11-slim
5. **Test locally first** before deploying

## Getting Help

1. **Check platform status pages**
2. **Review platform documentation**
3. **Search platform community forums**
4. **Contact platform support** with specific error messages
5. **Try alternative platforms** as backup

Remember: Docker deployment resolves 90% of Python environment issues by using pre-built, tested environments instead of compiling from source.